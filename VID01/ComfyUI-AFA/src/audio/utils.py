
from typing import Sequence, Mapping, Any, Union
from pathlib import Path
import torch
import json
import soundfile as sf
from pydub import AudioSegment


def list_wav_files(folder_path):
    return [str(p) for p in Path(folder_path).glob("*.wav")]

    
def find_first_available(folder, basename, width=3):
    folder = Path(folder)

    i = 0
    while True:
        name = f"{basename}{i:0{width}d}"
        path = folder / name
        if not path.exists():
            return str(path)
        i += 1

    raise Exception("find_first_available did not find a path")


def load_audio_sf(path: str):
    print("##################### reading file:", path)
    data, sr = sf.read(path, dtype="float32")

    # Ensure shape = [channels, samples]
    if data.ndim == 1:
        data = data[None, :]          # mono → [1, samples]
    else:
        data = data.T                 # [samples, ch] → [ch, samples]

    return {
        "waveform": torch.from_numpy(data),
        "sample_rate": sr,
    }


def get_actual_audio_duration(audio_path):
    try:
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0  # Convert milliseconds to seconds
    except ImportError:
        print("pydub not installed. Using default duration.")
        return 5.0
    except Exception as e:
        print(f"Could not read audio duration for {audio_path}: {e}")
        return 5.0
    

def read_segments(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data



def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]