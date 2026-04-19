import json
import os
from pathlib import Path
import opentimelineio as otio
from opentimelineio.opentime import TimeRange, RationalTime
import subprocess
import sys

from  .kdenlive import write_to_string
from .utils import get_actual_audio_duration

def create_speaker_timeline(segments, audio_folder=None):    
    segments.sort(key=lambda x: x['start'])
    
    # Dictionary to store tracks for each speaker
    speaker_tracks = {}
    
    # Dictionary to track end times for each speaker (for overlap handling)
    speaker_end_times = {}
    
    # Process each clip
    for clip_idx, clip in enumerate(segments):
        start_time = clip['start']
        filename = clip['ar_filename']
        
        # Extract speaker ID from filename
        # Assuming format: segment_X_SPEAKER_YY_ar.wav
        parts = filename.split('_')
        speaker_id = None
        for part_idx, part in enumerate(parts):
            if part.startswith('SPEAKER') and ((part_idx+1) < len(parts)):
                speaker_id = parts[part_idx+1]
                break
        
        if speaker_id is None:
            speaker_id = f"SPEAKER_UNKNOWN_{clip_idx}"
            print(f"Warning: Could not extract speaker ID from {filename}, using {speaker_id}")
        
        # Full path to audio file
        audio_path = os.path.join(audio_folder, filename)
        
        # Check if audio file exists
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found: {audio_path}")
            duration = 5.0  
        else:
            duration = get_actual_audio_duration(audio_path)
        
        # Create time objects
        # start_rt = RationalTime(start_time, 1.0)  # 1.0 means seconds
        duration_rt = RationalTime(duration, 1.0)
        
        # Handle overlaps: if this clip starts before previous clip ends, adjust previous clip
        if speaker_id in speaker_end_times:
            last_end = speaker_end_times[speaker_id]
            if start_time < last_end:
                # Overlap detected - we need to adjust the previous clip's duration
                # Find the track for this speaker
                if speaker_id in speaker_tracks:
                    track = speaker_tracks[speaker_id]
                    # Get the last clip in the track
                    if len(track) > 0:
                        last_clip = track[-1]
                        
                        # Calculate new duration to end at current clip's start
                        # new_duration = start_time - (last_clip.source_range.start_time.value + 
                        #                            last_clip.source_range.duration.value)
                        
                        new_duration = start_time - last_clip.source_range.start_time.value

                        if new_duration > 0:
                            # Update the duration of the last clip
                            last_clip.source_range = TimeRange(
                                start_time=last_clip.source_range.start_time,
                                duration=RationalTime(new_duration, 1.0)
                            )
                            print(f"Adjusted clip duration for {speaker_id}: {last_clip.name} -> {new_duration}s")
                        else:
                            # Remove the last clip if new duration is negative or zero
                            print(f"Removing overlapped clip for {speaker_id}: {last_clip.name}")
                            track.pop()
        
        # Create media reference for the audio file
        media_reference = otio.schema.ExternalReference(
            target_url=audio_path,
            available_range=TimeRange(
                start_time=RationalTime(0, 1.0),
                duration=duration_rt
            )
        )
        
        # Create the clip
        otio_clip = otio.schema.Clip(
            name=f"{speaker_id}_{clip_idx}",
            media_reference=media_reference,
            source_range=TimeRange(
                start_time=RationalTime(0, 1.0),
                duration=duration_rt
            )
        )
        
        # Get or create track for this speaker
        if speaker_id not in speaker_tracks:
            # Create a new audio track for this speaker
            track = otio.schema.Track(
                name=f"SPKR_{speaker_id}",
                kind=otio.schema.TrackKind.Audio
            )
            speaker_tracks[speaker_id] = track
        
        # Add clip to the speaker's track at the correct time
        # speaker_tracks[speaker_id].append(otio_clip)




        track = speaker_tracks[speaker_id]
        # Calculate current end time of the track
        current_track_time = 0.0
        for item in track:
            if item.source_range:
                current_track_time += item.source_range.duration.value

        # If there is a gap between last clip end and this clip start, insert a Gap
        gap_duration = start_time - current_track_time

        if gap_duration > 0:
            gap = otio.schema.Gap(
                source_range=TimeRange(
                    start_time=RationalTime(0, 1.0),
                    duration=RationalTime(gap_duration, 1.0)
                )
            )
            track.append(gap)

        # Now append the clip
        track.append(otio_clip)






        
        # Update end time for this speaker
        speaker_end_times[speaker_id] = start_time + duration
    
    # Create timeline
    timeline = otio.schema.Timeline(audio_folder)
    
    # Add all speaker tracks to the timeline
    for speaker_id, track in speaker_tracks.items():
        timeline.tracks.append(track)
    
    return timeline


def process_json_to_kdenlive(json_path, output_folder=None):
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)
    
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return
    
    if output_folder is None:
        output_folder = json_path.parent
    
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Create timeline
    print(f"Processing {json_path}...")
    timeline = create_speaker_timeline(segments, str(output_folder))
    
    # Save as OTIO JSON
    otio_path = output_folder / f"{json_path.stem}_timeline.otio"
    otio.adapters.write_to_file(timeline, str(otio_path))
    print(f"Saved OTIO timeline to: {otio_path}")
    
    # Convert to Kdenlive format using otioconvert
    kdenlive_path = output_folder / f"{json_path.stem}_project.kdenlive"
    
    try:
        # Method 1: Direct OTIO adapter write
        str_kdenlive = write_to_string(timeline)
        with open(kdenlive_path, "w", encoding="utf-8") as f:
            f.write(str_kdenlive)
        print(f"Saved Kdenlive project to: {kdenlive_path}")
    except Exception as e:
        print(f"Direct conversion failed: {e}")
        print("Trying alternative method with otioconvert...")
    
    # Print summary
    print("\n=== Timeline Summary ===")
    print(f"Timeline: {timeline.name}")
    print(f"Duration: {timeline.duration().value} seconds")
    print(f"Number of tracks: {len(timeline.tracks)}")
    
    for i, track in enumerate(timeline.tracks):
        print(f"  Track {i}: {track.name}")
        print(f"    Number of clips: {len(track)}")
        print(f"    Track duration: {track.duration().value} seconds")
    
    return timeline, str(kdenlive_path)



if __name__ == "__main__":
    

    json_file = r"diarization.json"
    
    # Process the JSON file
    timeline, kdenlive_path = process_json_to_kdenlive(json_file)
    
    print(f"\nDone! You can open {kdenlive_path} in Kdenlive.")