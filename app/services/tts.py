import torch
import asyncio
import soundfile as sf  
import time 
from ..core.celery_app import celery_app
import io

_models = {}


async def _load_tts_model_async(language='ru', device=torch.device('cuda')):
    model_key = (language, str(device))
    if model_key not in _models:
        model_id = 'v4_ru' if language == 'ru' else 'v3_en'
        model, _ = await asyncio.to_thread(
            torch.hub.load,
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language=language,
            speaker=model_id
        )
        model.to(device)
        _models[model_key] = model
    return _models[model_key]

def load_tts_model_sync(language='ru', device=torch.device('cuda')):
    return asyncio.run(_load_tts_model_async(language, device))

@celery_app.task(ignore_result=False) 
def silero_tts_task(
    text: str,
    language: str = 'ru',
    speaker: str = 'xenia',
    sample_rate: int = 48000,
    put_accent: bool = True,
    put_yo: bool = True,
    device_str: str = 'cuda',
) -> tuple[str, bytes]:
    """
    Celery task to generate speech from text using Silero TTS.
    Returns tuple: (original_text, audio_bytes)
    """
    language = language.lower()
    if language not in ['ru', 'en']:
        raise ValueError("Language must be either 'ru' or 'en'")

    if language == 'en' and sample_rate == 48000:
        sample_rate = 24000

    device = torch.device(device_str)

    model = load_tts_model_sync(language, device)
    
    if language == 'ru':
        audio_tensor = model.apply_tts(
            text=text,
            speaker=speaker,
            sample_rate=sample_rate,
            put_accent=put_accent,
            put_yo=put_yo
        )
    else:
        audio_tensor = model.apply_tts(
            text=text,
            speaker="en_0",
            sample_rate=sample_rate
        )
    
    audio_numpy = audio_tensor.cpu().numpy()
    buffer = io.BytesIO()
    sf.write(
        buffer,
        audio_numpy,
        sample_rate,
        format='WAV'
    )
    audio_bytes = buffer.getvalue()

    return text, audio_bytes

async def main_async():
    """Example async usage for testing tasks (requires running worker)"""
    ru_text = 'В недрах тундры выдры в гетрах тырят в вёдра ядра кедров.'
    en_text = 'The quick brown fox jumps over the lazy dog.'

    print("Submitting Russian TTS task...")
    ru_task = silero_tts_task.delay(ru_text, language='ru')
    try:
        original_text, ru_audio_bytes = ru_task.get(timeout=60) 
        print(f"Russian TTS task completed, audio size: {len(ru_audio_bytes)} bytes")
        with open("russian_output_task.wav", "wb") as f:
            f.write(ru_audio_bytes)
    except Exception as e:
        print(f"Russian TTS task failed: {e}")

    print("\nSubmitting English TTS task...")
    en_task = silero_tts_task.delay(en_text, language='en', sample_rate=24000)
    try:
        original_text, en_audio_bytes = en_task.get(timeout=60) 
        print(f"English TTS task completed, audio size: {len(en_audio_bytes)} bytes")
        with open("english_output_task.wav", "wb") as f:
            f.write(en_audio_bytes)
    except Exception as e:
        print(f"English TTS task failed: {e}")