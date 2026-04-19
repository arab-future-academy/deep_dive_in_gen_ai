import os
from typing import Any, Dict, List
import json
import torch
from nodes import NODE_CLASS_MAPPINGS  # Import for registering custom nodes
import soundfile as sf
import numpy as np
import torch


from .gen_arabic_speech import generate_arabic_speech
from .utils import read_segments


class AFAArabicTTS:
    CATEGORY = "AFA"

    def __init__(self):
        # Node metadata
        ...

    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "out_folder": ("STRING",),
                "seed": ("INT",),
            }
        }


    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("segments_speech", "out_folder")

    FUNCTION = "gen_audio"

    def gen_audio(self, out_folder: str, seed: int) -> tuple[List[Dict[str, Any]], str]:
        json_diarization_path = out_folder + "/diarization.json"
        segments = read_segments(json_diarization_path)
        
        for seg in segments:
            print(seg)
            wav_path = out_folder + "/" + seg["filename"]
            print(wav_path)
            arabic_audio = generate_arabic_speech(seg["ar"], seed, out_folder + "/" + seg["filename"])
            
            waveform = arabic_audio["waveform"]
            sample_rate = arabic_audio["sample_rate"]
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.tensor(waveform)
            waveform = waveform.cpu()
            waveform = waveform.numpy()

            # Ensure shape is (channels, time)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.ndim == 3:
                waveform = waveform.squeeze(0)
            
            waveform = waveform.T

            ar_filename = seg["filename"][:-4] + "_ar.wav"
            filename = out_folder + "/" + ar_filename
            folder = os.path.dirname(filename)
            os.makedirs(folder, exist_ok=True)
            sf.write(filename, waveform, sample_rate)
            print(f"Saved: {filename}")

            seg["ar_filename"] = ar_filename


        with open(json_diarization_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=4)
                        



        return (segments, out_folder)  
