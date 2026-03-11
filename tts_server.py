"""
TTS Server - Lightweight FastAPI server for Text-to-Speech
Supports English (en), Telugu (te), and Hindi (hi) via gTTS.
Frontend calls: GET /speak?text=<text>&lang=<en|te|hi>
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from gtts import gTTS
import io

app = FastAPI(title="TTS Server", version="1.0.0")

# Allow all origins so the kiosk frontend can reach this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Map frontend lang codes → gTTS lang codes
LANG_MAP = {
    "en": "en",
    "te": "te",   # Telugu
    "hi": "hi",   # Hindi
}


@app.get("/speak")
def speak(
    text: str = Query(..., min_length=1, description="Text to synthesize"),
    lang: str = Query("en", description="Language code: en | te | hi"),
):
    gtts_lang = LANG_MAP.get(lang)
    if not gtts_lang:
        raise HTTPException(status_code=400, detail=f"Unsupported lang '{lang}'. Use: en, te, hi")

    try:
        tts = gTTS(text=text, lang=gtts_lang)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {e}")

    return StreamingResponse(buf, media_type="audio/mpeg")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)


# uvicorn tts_server:app --host 0.0.0.0 --port 5000 --reload