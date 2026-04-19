import os
from typing import Any, Dict, List

import json
import torch

from nodes import NODE_CLASS_MAPPINGS  

import soundfile as sf
import numpy as np
import torch
from pydub import AudioSegment

from .gen_arabic_speech import generate_arabic_speech
from .export_otio import process_json_to_kdenlive


# random.randint(1, 2**64)

from pathlib import Path


def read_segments(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def export_one_audio(segments, base_dir, out_path):
    # Determine total length
    total_duration_ms = max(int(seg["end"] * 1000) for seg in segments)

    # Start with silence
    final_audio = AudioSegment.silent(duration=total_duration_ms)

    # Overlay each segment at correct position
    for seg in segments:
        audio = AudioSegment.from_file(base_dir + "/" + seg["ar_filename"])
        start_ms = int(seg["start"] * 1000)
        final_audio = final_audio.overlay(audio, position=start_ms)

    # Export merged file
    final_audio.export(out_path, format="wav")
    print("Merged WAV saved as merged_arabic.wav")

    final_audio = final_audio.set_channels(1).set_sample_width(2)

    samples = np.array(final_audio.get_array_of_samples()).astype(np.float32)
    samples /= np.iinfo(np.int16).max

    waveform = torch.from_numpy(samples).unsqueeze(0).unsqueeze(0)

    audio_output = {
        "waveform": waveform,
        "sample_rate": final_audio.frame_rate
    }

    return audio_output



class AFAExport:
    CATEGORY = "AFA"

    def __init__(self):
        # Node metadata
        ...

    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "out_folder": ("STRING",),
            }
        }


    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "segments_speech", "out_folder")

    FUNCTION = "export_data"

    def export_data(self, out_folder: str) -> tuple[List[Dict[str, Any]], str]:
        print("-------------------########## GEN AUDIO 01 ############-----------------")
        json_diarization_path = out_folder + "/diarization.json"
        segments = read_segments(json_diarization_path)
        
        timeline, kdenlive_path = process_json_to_kdenlive(json_diarization_path)
        # generate_kdenlive_project(segments)
        audio_output = export_one_audio(segments, out_folder, out_folder + "/translated.wav")



        return (audio_output, segments, out_folder)  
