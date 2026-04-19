import os
from pyannote.audio import Pipeline
from typing import Any, Dict, List
import json
import torch
from nodes import NODE_CLASS_MAPPINGS 
import folder_paths
import soundfile as sf
import torch
from .utils import find_first_available




class AFASpeakerSeparate:
    CATEGORY = "AFA"

    def __init__(self):
        # Node metadata
        ...

    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "audio": ("AUDIO",),
                "hf_token": ("STRING",), 
                "file_name_prefix": ("STRING",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("speaker_segments", "out_folder", "sample_rate")

    FUNCTION = "separate_audio"

    def separate_audio(self, audio: dict, hf_token: str, file_name_prefix: str) -> List[Dict[str, Any]]:

        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
        
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        # Ensure tensor
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform)

        waveform = waveform.cpu()

        # Ensure shape is (channels, time)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim == 3:
            waveform = waveform.squeeze(0)

        file = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }


        diarization = pipeline(file)
        
        speaker_segments = []
        i = 0
        for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
            print( f"start : {turn.start}, end: {turn.end}, speaker: {speaker}")
            speaker_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
                "filename":  f"segment_{i+1}_{speaker}.wav"
            })

            i += 1
        
        
                
                
        filename = f"diarization.json"            
        print(folder_paths.get_output_directory())

        out_folder = find_first_available(folder_paths.get_output_directory(), file_name_prefix)
        print("out_folder=", out_folder)

        filename = out_folder  + "/" + filename 
        print("filename=", filename)
        folder = os.path.dirname(filename)
        os.makedirs(folder, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(speaker_segments, f, indent=4)

        audio_np = waveform.numpy().astype("float32")
        # Create separate audio files for each segment
        for i, seg in enumerate(speaker_segments):
            start_sample = int(seg["start"] * sample_rate)
            end_sample = int(seg["end"] * sample_rate)

            # Slice audio: shape (channels, samples)
            segment_audio = audio_np[:, start_sample:end_sample]

            # Transpose to (samples, channels) for soundfile
            segment_audio = segment_audio.T


            filename = seg["filename"]
            filename = out_folder + "/" + filename 
            folder = os.path.dirname(filename)
            os.makedirs(folder, exist_ok=True)


            sf.write(filename, segment_audio, sample_rate)
            print(f"Saved: {filename}")
            



        return (speaker_segments, out_folder, sample_rate)  