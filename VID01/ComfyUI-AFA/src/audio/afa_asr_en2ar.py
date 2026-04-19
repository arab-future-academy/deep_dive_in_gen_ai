
from typing import Any, Dict, List
import json
from nodes import NODE_CLASS_MAPPINGS  # Import for registering custom nodes
from transformers import MarianMTModel, MarianTokenizer

import whisper

from .utils import list_wav_files, read_segments



class AFAAsrAr2En:
    CATEGORY = "AFA"

    def __init__(self):
        # Node metadata
        self.asr_model = None
        self.en2ar_model = None
        self.en2ar_tokenizer = None

    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "folder": ("STRING",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("segments_en_ar", "out_folder")

    FUNCTION = "text_rec"

    def text_rec(self, folder: str) -> tuple[List[Dict[str, Any]], str]:
        if not self.asr_model:
            self.asr_model = whisper.load_model("base")
        if not self.en2ar_model:
            model_name = "Helsinki-NLP/opus-mt-en-ar"
            self.en2ar_tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.en2ar_model = MarianMTModel.from_pretrained(model_name)
             
        json_diarization_path = folder + "/diarization.json"
        segments = read_segments(json_diarization_path)
        
        for seg in segments:
            wav_path = folder + "/" + seg["filename"]
            result = self.asr_model.transcribe(wav_path)
            seg["en"] = result["text"]

            print("-------- en: ", seg["en"])

            tokens = self.en2ar_tokenizer(seg["en"], return_tensors="pt", padding=True)
            translated = self.en2ar_model.generate(**tokens)
            seg["ar"] = self.en2ar_tokenizer.decode(translated[0], skip_special_tokens=True)

            print("-------- en: ", seg["ar"])



        with open(json_diarization_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=4)
        
                 



        return (segments, folder)