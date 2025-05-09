from fastapi import FastAPI, UploadFile, Depends, HTTPException, status, Form
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from .core.db import Base, engine, SessionLocal
from .core.models import User, UserMessage
from .auth.auth import get_db, authenticate_user, create_access_token, get_current_user, get_password_hash
from .core.billing import charge_credits
from .services.asr import transcribe_audio_task
from .services.llm import get_llm_response_task
from .services.tts import silero_tts_task
from math import ceil
import io
import wave
import logging
from prometheus_fastapi_instrumentator import Instrumentator


from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


# Setup logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base.metadata.create_all(bind=engine)
app = FastAPI()
Instrumentator().instrument(app).expose(app)


app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/register/")
def register(email: str = Form(...), password: str = Form(...), lang: str = Form("ru"), db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(email=email, hashed_password=get_password_hash(password), lang=lang)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"msg": "User registered"}

@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/voice-query/")
async def voice_query(audio: UploadFile, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    logger.info(f"Received voice query from user {user.email}")
    
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file received.")

    try:
        logger.info("Submitting ASR task")
        asr_task = transcribe_audio_task.apply_async(
            args=[audio_bytes],
            queue='asr_priority'
        )

        text, user_audio_bytes_from_task, detected_lang = asr_task.get() 
        logger.info(f"ASR task completed. Detected lang: {detected_lang}, Text: {text[:50]}...")
        lang = detected_lang if detected_lang in ("ru", "en") else user.lang

        # Сохраняем сообщение пользователя
        user_msg = UserMessage(user_id=user.id, role="user", content=text, lang=lang)
        db.add(user_msg)
        db.commit()
        db.refresh(user_msg)
        
        history = db.query(UserMessage).filter_by(user_id=user.id).order_by(UserMessage.timestamp.desc()).limit(5).all()
        history = list(reversed(history))
        
        messages = []
        for msg in history:
            role = "user" if msg.role == "user" else "assistant"
            messages.append({"role": role, "content": msg.content})
        if not messages or messages[-1]["content"] != text:
             messages.append({"role": "user", "content": text})

        # 2. LLM Task
        logger.info("Submitting LLM task")
        llm_task = get_llm_response_task.delay(messages, lang=lang)
        llm_response = llm_task.get(timeout=120) 
        logger.info("LLM task completed.")

        # Парсим ответ LLM
        answer, correction = "", ""
        if lang == "ru":
            if "Ответ:" in llm_response:
                after_answer = llm_response.split("Ответ:",1)[1]
                if "Исправление:" in after_answer:
                    answer = after_answer.split("Исправление:",1)[0].strip()
                    correction = after_answer.split("Исправление:",1)[1].strip()
                else:
                    answer = after_answer.strip()
                    correction = ""
            else:
                answer = llm_response.strip()
                correction = ""
        else:
            if "Answer:" in llm_response and "Correction:" in llm_response:
                answer = llm_response.split("Answer:",1)[1].split("Correction:",1)[0].strip()
                correction = llm_response.split("Correction:",1)[1].strip()
            elif "Answer:" in llm_response:
                 answer = llm_response.split("Answer:",1)[1].strip()
                 correction = "" 
            else:
                answer = llm_response.strip()
                correction = ""
        
        # Сохраняем ответы ассистента
        db.add(UserMessage(user_id=user.id, role="assistant", content=answer, lang=lang))
        if correction:
            db.add(UserMessage(user_id=user.id, role="assistant", content=correction, lang=lang))
        db.commit()

        # 3. TTS Task
        logger.info("Submitting TTS task")
        tts_task = silero_tts_task.delay(text=answer, language=lang)
        _, answer_audio_bytes = tts_task.get(timeout=120)
        logger.info("TTS task completed.")
        
        def get_duration_from_bytes(wav_bytes):
            if not wav_bytes:
                return 0
            try:
                with io.BytesIO(wav_bytes) as f:
                    with wave.open(f, 'rb') as wf:
                        frames = wf.getnframes()
                        rate = wf.getframerate()
                        return frames / float(rate) if rate > 0 else 0
            except (wave.Error, EOFError, ValueError) as e:
                logger.error(f"Could not get duration from WAV bytes: {e}")
                return 0

        user_audio_duration = get_duration_from_bytes(audio_bytes)
        answer_audio_duration = get_duration_from_bytes(answer_audio_bytes)
        total_seconds = user_audio_duration + answer_audio_duration
        total_minutes = ceil(total_seconds / 60)
        
        logger.info(f"Billing: User audio: {user_audio_duration:.2f}s, Answer audio: {answer_audio_duration:.2f}s, Total mins: {total_minutes}")
        charge_credits(user, db, amount=5 * total_minutes)
        logger.info(f"Charged {5 * total_minutes} credits. User {user.email} remaining credits: {user.credits}")

        return {
            "text": text,
            "answer": answer,
            "correction": correction,
            "answer_audio": answer_audio_bytes.hex(),
            "charged_minutes": total_minutes,
            "charged_rub": 5 * total_minutes,
            "lang": lang
        }

    except Exception as e:
        logger.error(f"Error processing voice query for user {user.email}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/me/")
def me(user: User = Depends(get_current_user)):
    return {"email": user.email, "credits": user.credits, "lang": user.lang}

@app.get("/", response_class=FileResponse)
async def root():
    return FileResponse("app/static/index.html")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)