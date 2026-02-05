from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import librosa
import numpy as np
import requests
import tempfile

# -----------------------
# APP INIT
# -----------------------
app = FastAPI(title="AI Generated Voice Detection API")

# -----------------------
# CONFIG
# -----------------------
API_KEY = "my_secret_key_123"

# -----------------------
# REQUEST MODEL
# -----------------------
class AudioURLRequest(BaseModel):
    audio_url: str
    message: str | None = None

# -----------------------
# VOICE ANALYSIS LOGIC
# -----------------------
def analyze_voice(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]

    if len(pitch_values) == 0:
        return "Human", 0.55, "No artificial pitch pattern detected"

    pitch_std = np.std(pitch_values)

    if pitch_std < 20:
        return (
            "AI-generated",
            0.85,
            "Pitch variation is very low, indicating synthetic voice"
        )
    else:
        return (
            "Human",
            0.90,
            "Natural pitch variation detected"
        )

# -----------------------
# API ENDPOINT
# -----------------------
@app.post("/predict")
async def predict_voice(request: Request, data: AudioURLRequest):

    # 🔐 AUTH CHECK (RAW HEADER)
    auth_header = request.headers.get("authorization")

    if auth_header != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # DOWNLOAD AUDIO
    try:
        response = requests.get(data.audio_url, timeout=10)
        response.raise_for_status()
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to download audio file")

    # SAVE TEMP FILE
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
        temp_audio.write(response.content)
        audio_path = temp_audio.name

    classification, confidence, explanation = analyze_voice(audio_path)

    return {
        "classification": classification,
        "confidence_score": round(confidence, 2),
        "explanation": explanation
    }
