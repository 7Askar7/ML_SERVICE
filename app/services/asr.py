from faster_whisper import WhisperModel
import tempfile
import logging
from ..core.celery_app import celery_app

logger = logging.getLogger(__name__)

@celery_app.task(bind=True)
def transcribe_audio_task(self, audio_bytes: bytes):
    """Celery task to transcribe audio from bytes"""
    logger.info("ASR: start loading model")
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    logger.info("ASR: model loaded")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        
        logger.info("ASR: before model inference")
        # Transcribe the audio
        segments, info = model.transcribe(tmp.name, beam_size=5)
        text = " ".join([segment.text for segment in segments])
        logger.info("ASR: after model inference")
    
    detected_lang = info.language if hasattr(info, 'language') else None

    return text, audio_bytes, detected_lang
