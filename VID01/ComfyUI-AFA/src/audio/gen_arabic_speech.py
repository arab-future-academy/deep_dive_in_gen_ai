
import os
import sys
    
from nodes import NODE_CLASS_MAPPINGS

import soundfile as sf
import torch

from .utils import get_value_at_index, load_audio_sf



def generate_arabic_speech(arabic_text: str, seed: int, voice_to_clone: str):
    # # import_custom_nodes()
    with torch.inference_mode():
        primitivestringmultiline = NODE_CLASS_MAPPINGS["PrimitiveStringMultiline"]()
        primitivestringmultiline_43 = primitivestringmultiline.EXECUTE_NORMALIZED(
            value=arabic_text
        )

        chatterboxofficial23langenginenode = NODE_CLASS_MAPPINGS[
            "ChatterBoxOfficial23LangEngineNode"
        ]()
        chatterboxofficial23langenginenode_73 = (
            chatterboxofficial23langenginenode.create_engine_adapter(
                model_version="v2",
                language="Arabic",
                device="auto",
                exaggeration=0.5,
                temperature=0.8,
                cfg_weight=0.5,
                repetition_penalty=2,
                min_p=0.05,
                top_p=1,
            )
        )

        charactervoicesnode = NODE_CLASS_MAPPINGS["CharacterVoicesNode"]()
        unifiedttstextnode = NODE_CLASS_MAPPINGS["UnifiedTTSTextNode"]()
        previewany = NODE_CLASS_MAPPINGS["PreviewAny"]()
        # saveaudio = NODE_CLASS_MAPPINGS["SaveAudio"]()

        for q in range(1):
            charactervoicesnode_51 = charactervoicesnode.get_voice_reference(
                voice_name=None, 
                reference_text="",
                opt_audio_input=load_audio_sf(voice_to_clone),
            )

            unifiedttstextnode_47 = unifiedttstextnode.generate_speech(
                text=get_value_at_index(primitivestringmultiline_43, 0),
                narrator_voice="none",
                seed=seed,
                enable_chunking=True,
                max_chars_per_chunk=400,
                chunk_combination_method="auto",
                silence_between_chunks_ms=100,
                enable_audio_cache=True,
                batch_size=0,
                TTS_engine=get_value_at_index(chatterboxofficial23langenginenode_73, 0),
                opt_narrator=get_value_at_index(charactervoicesnode_51, 0),
            )

            previewany_50 = previewany.main(
                source=get_value_at_index(unifiedttstextnode_47, 1)
            )

            return get_value_at_index(unifiedttstextnode_47, 0)
        


