
from .src.misc.afa_calculator import AFACalc
from .src.image.afa_image_filter import AFAImageFilter
from .src.audio.afa_speaker_separation import AFASpeakerSeparate
from .src.audio.afa_asr_en2ar import AFAAsrAr2En
from .src.audio.afa_arabic_tts import AFAArabicTTS
from .src.audio.afa_export_audio import AFAExport



NODE_CLASS_MAPPINGS = {
    "AFA Calc" : AFACalc,
    "AFA Image Filter": AFAImageFilter,
    "AFA Speaker Separation": AFASpeakerSeparate,
    "AFA ASR Translation": AFAAsrAr2En,
    "AFA Arabic TTS": AFAArabicTTS,
    "AFA Export": AFAExport
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AFA Calc": "عقدة الحساب",
    "AFA Image Filter": "عمليات على الصورة",
    "AFA Speaker Separation": "Separate Speakers",
    "AFA ASR Translation": "EnglishText and Translation",
    "AFA Arabic TTS": "Generate Arabic Audio",
    "AFA Export": "Export Audio & Kdenlive"
}