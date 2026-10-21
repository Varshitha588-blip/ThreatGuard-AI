import os
import tempfile
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from app.services.text_detector import predict_text
from app.services.audio_detector import predict_audio
from app.services.explanation import explain_text, explain_audio

app = FastAPI(
    title="AI Phishing and Deepfake Detection API",
    version="0.2.0",
)
# Keep local development origins and allow the deployed frontend origin
# to be configured through the CORS_ORIGINS environment variable.
_default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
_extra_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]
_allowed_origins = list(dict.fromkeys(_default_origins + _extra_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


@app.get("/")
def root():
    return {
        "message": "Threat Detection API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze/text")
def analyze_text(request: TextRequest):
    try:
        result = predict_text(request.text)
        return {
            "input_type": "text",
            **result,
            "explanation": explain_text(request.text, result),
            "warning": (
                "Experimental model; predictions are not proof of maliciousness."
            ),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/analyze/audio")
async def analyze_audio(file: UploadFile = File(...)):
    allowed_extensions = {".wav", ".mp3", ".flac", ".ogg"}
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file extension. Upload WAV, MP3, FLAC, or OGG.",
        )

    temp_path = None

    try:
        # Limit upload size to 20 MB.
        max_bytes = 20 * 1024 * 1024
        contents = await file.read(max_bytes + 1)

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="The uploaded audio file is empty.",
            )

        if len(contents) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail="Audio file exceeds the 20 MB upload limit.",
            )

        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        result = predict_audio(temp_path)

        return {
            "input_type": "audio",
            "filename": Path(filename).name,
            **result,
            "explanation": explain_audio(result),
            "warning": (
                "Experimental model trained on a small dataset. "
                "Predictions are not proof that a recording is genuine or spoofed."
            ),
        }

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        await file.close()
