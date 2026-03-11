"""
TTS + STT Server — FastAPI  (v4 — Vosk OR Whisper, switchable via .env)
────────────────────────────────────────────────────────────────────────
TTS : GET  /speak?text=<text>&lang=<en|te|hi>  →  audio/mpeg  (gTTS)
STT : POST /stt  (multipart: audio=<file> + lang=<en|te|hi>)  →  {"text":"..."}

Engine is chosen by the STT_ENGINE environment variable (default: whisper):
  STT_ENGINE=whisper   →  openai-whisper  (offline, high accuracy)
  STT_ENGINE=vosk      →  Vosk            (offline, fast, tiny models)

Whisper model size:  WHISPER_MODEL=base  (tiny|base|small|medium|large)

Vosk model layout:
  models/vosk-model-en/
  models/vosk-model-te/
  models/vosk-model-hi/

Run:  uvicorn tts_server:app --host 0.0.0.0 --port 5000 --reload
"""

import io
import json
import os
import subprocess
import tempfile
import wave

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from gtts import gTTS

# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

STT_ENGINE    = os.getenv("STT_ENGINE", "whisper").lower().strip()
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base").strip()

# ── Locate ffmpeg binary (imageio-ffmpeg bundle, then system PATH) ─────────────
def _find_ffmpeg() -> str:
    """Return an absolute path to an ffmpeg executable, or raise RuntimeError."""
    # 1. Try imageio-ffmpeg bundled binary (no system install needed)
    try:
        import imageio_ffmpeg as _iio
        path = _iio.get_ffmpeg_exe()
        if os.path.isfile(path):
            return path
    except Exception:
        pass

    # 2. Fall back to system PATH
    import shutil
    path = shutil.which("ffmpeg")
    if path:
        return path

    raise RuntimeError(
        "ffmpeg not found. Run:  pip install imageio-ffmpeg\n"
        "  OR install ffmpeg system-wide and add it to PATH."
    )

try:
    FFMPEG_BIN = _find_ffmpeg()
    # Add ffmpeg's directory to PATH so Whisper's internal subprocess call finds it too
    _ffmpeg_dir = os.path.dirname(FFMPEG_BIN)
    os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    print(f"  ✓ ffmpeg: {FFMPEG_BIN}")
except RuntimeError as _e:
    FFMPEG_BIN = ""
    print(f"  ✗ {_e}")


# ── Audio conversion via direct ffmpeg subprocess (no pydub needed) ────────────
def convert_to_wav_16k(audio_bytes: bytes, suffix: str = ".webm") -> bytes:
    """
    Convert any audio format (webm, ogg, mp4, …) to 16 kHz mono 16-bit PCM WAV
    using the imageio-ffmpeg bundled binary.  No pydub, no system ffmpeg required.
    """
    if not FFMPEG_BIN:
        raise RuntimeError("No ffmpeg binary available for audio conversion.")

    # Write input to a temp file (ffmpeg needs seekable input for webm/ogg)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f_in:
        f_in.write(audio_bytes)
        in_path = f_in.name

    out_path = in_path + ".wav"

    cmd = [
        FFMPEG_BIN, "-y",
        "-i", in_path,
        "-ar", "16000",
        "-ac", "1",
        "-sample_fmt", "s16",
        "-f", "wav",
        out_path,
    ]
    print(f"  [ffmpeg] running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)

        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="replace")
            out = result.stdout.decode("utf-8", errors="replace")
            print(f"\n=== ffmpeg FAILED (rc={result.returncode}) ===\nSTDERR: {err}\nSTDOUT: {out}\n====================\n")
            raise RuntimeError(f"ffmpeg conversion failed (rc={result.returncode}): {err[:300]}")

        with open(out_path, "rb") as f_out:
            return f_out.read()

    finally:
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


# ── Whisper ────────────────────────────────────────────────────────────────────
_whisper_model = None
WHISPER_AVAILABLE = False

if STT_ENGINE == "whisper":
    try:
        import whisper as _whisper_lib
        WHISPER_AVAILABLE = True
    except ImportError:
        print("⚠  openai-whisper not installed — run: pip install openai-whisper")

WHISPER_LANG = {"en": "english", "te": "telugu", "hi": "hindi"}


# ── Vosk ───────────────────────────────────────────────────────────────────────
_vosk_models: dict = {}
VOSK_AVAILABLE = False

if STT_ENGINE == "vosk":
    try:
        from vosk import KaldiRecognizer, Model as VoskModel
        VOSK_AVAILABLE = True
    except ImportError:
        print("⚠  vosk not installed — run: pip install vosk")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="VIET TTS+STT Server", version="4.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

VOSK_LANG_DIRS = {
    "en": "vosk-model-en",
    "te": "vosk-model-te",
    "hi": "vosk-model-hi",
}


def _load_whisper():
    global _whisper_model
    if not WHISPER_AVAILABLE:
        return
    print(f"  Loading Whisper model: {WHISPER_MODEL} …")
    # Honour XDG_CACHE_HOME so the model loads from the path baked in at build time
    cache_dir = os.getenv("XDG_CACHE_HOME")
    if cache_dir:
        download_root = os.path.join(cache_dir, "whisper")
        os.makedirs(download_root, exist_ok=True)
    else:
        download_root = None  # whisper uses its default ~/.cache/whisper
    try:
        _whisper_model = _whisper_lib.load_model(WHISPER_MODEL, download_root=download_root)
        print(f"  ✓ Whisper [{WHISPER_MODEL}] ready")
    except Exception as exc:
        print(f"  ✗ Whisper load failed: {exc}")


def _load_vosk():
    if not VOSK_AVAILABLE:
        return
    for lang, folder in VOSK_LANG_DIRS.items():
        path = os.path.join(MODEL_DIR, folder)
        if os.path.isdir(path):
            print(f"  Loading Vosk [{lang}] from {path}")
            try:
                _vosk_models[lang] = VoskModel(path)
                print(f"  ✓ Vosk [{lang}] ready")
            except Exception as exc:
                print(f"  ✗ Vosk [{lang}] failed: {exc}")
        else:
            print(f"  – Vosk [{lang}] model not found at {path}")


@app.on_event("startup")
def _startup():
    print(f"=== VIET Navigation Bot — TTS+STT Server v4.1 | engine: {STT_ENGINE.upper()} ===")
    if STT_ENGINE == "whisper":
        _load_whisper()
    elif STT_ENGINE == "vosk":
        _load_vosk()
    else:
        print(f"⚠  Unknown STT_ENGINE='{STT_ENGINE}'. Use 'whisper' or 'vosk' in backend/.env")


# ── TTS endpoint ───────────────────────────────────────────────────────────────
LANG_MAP_TTS = {"en": "en", "te": "te", "hi": "hi"}


@app.get("/speak")
def speak(
    text: str = Query(..., min_length=1),
    lang: str = Query("en"),
):
    gtts_lang = LANG_MAP_TTS.get(lang)
    if not gtts_lang:
        raise HTTPException(400, f"Unsupported lang '{lang}'. Use: en, te, hi")
    try:
        tts = gTTS(text=text, lang=gtts_lang)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
    except Exception as exc:
        raise HTTPException(500, f"TTS failed: {exc}")
    return StreamingResponse(buf, media_type="audio/mpeg")


# ── STT: Whisper transcription ─────────────────────────────────────────────────
def _transcribe_whisper(audio_bytes: bytes, lang: str) -> str:
    if _whisper_model is None:
        raise RuntimeError("Whisper model not loaded.")

    # Convert webm/ogg → 16 kHz WAV using bundled ffmpeg
    try:
        wav_bytes = convert_to_wav_16k(audio_bytes)
    except Exception as conv_err:
        raise RuntimeError(f"Audio conversion failed: {conv_err}")

    # Step 2: decode WAV PCM → float32 numpy array.
    # Passing numpy directly to Whisper bypasses its internal load_audio()
    # which would call bare 'ffmpeg' and fail with WinError 2.
    import numpy as np
    try:
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf) as wf:
            raw_pcm = wf.readframes(wf.getnframes())
        audio_np = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception as dec_err:
        raise RuntimeError(f"WAV decode failed: {dec_err}")

    # Step 3: transcribe numpy array — no ffmpeg call by Whisper
    result = _whisper_model.transcribe(
        audio_np,
        language=WHISPER_LANG.get(lang, "english"),
        fp16=False,
        verbose=False,
    )
    return result["text"].strip()


# ── STT: Vosk transcription ────────────────────────────────────────────────────
def _transcribe_vosk(audio_bytes: bytes, lang: str) -> str:
    model = _vosk_models.get(lang) or _vosk_models.get("en")
    if model is None:
        raise RuntimeError("No Vosk model available.")

    # Convert to 16 kHz mono WAV
    try:
        wav_bytes = convert_to_wav_16k(audio_bytes)
    except Exception as conv_err:
        raise RuntimeError(f"Audio conversion failed: {conv_err}")

    buf = io.BytesIO(wav_bytes)
    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(False)
    with wave.open(buf) as wf:
        while True:
            frames = wf.readframes(4000)
            if not frames:
                break
            rec.AcceptWaveform(frames)
    result = json.loads(rec.FinalResult())
    return result.get("text", "").strip()


# ── /stt endpoint ──────────────────────────────────────────────────────────────
@app.post("/stt")
async def stt(
    audio: UploadFile = File(..., description="Audio recording (webm/ogg/wav)"),
    lang:  str        = Form("en", description="Language: en | te | hi"),
):
    """Offline Speech-to-Text. Engine chosen by STT_ENGINE in backend/.env"""
    raw = await audio.read()
    if not raw:
        raise HTTPException(400, "Empty audio file received.")

    if not FFMPEG_BIN:
        raise HTTPException(
            503,
            "ffmpeg not available on this server. "
            "Run:  pip install imageio-ffmpeg  then restart."
        )

    try:
        if STT_ENGINE == "whisper":
            if not WHISPER_AVAILABLE or _whisper_model is None:
                raise HTTPException(503, "Whisper model not loaded. Check server logs.")
            text = _transcribe_whisper(raw, lang)

        elif STT_ENGINE == "vosk":
            if not VOSK_AVAILABLE or not _vosk_models:
                raise HTTPException(503, "Vosk model not loaded. Run: python download_model.py")
            text = _transcribe_vosk(raw, lang)

        else:
            raise HTTPException(500, f"Unknown STT_ENGINE '{STT_ENGINE}' in backend/.env")

    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        print(f"\n=== /stt ERROR ===\n{tb}==================\n")
        raise HTTPException(500, f"STT error: {exc}")

    return {"text": text, "lang": lang, "engine": STT_ENGINE}


# ── /health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":        "ok",
        "stt_engine":    STT_ENGINE,
        "ffmpeg":        FFMPEG_BIN or "NOT FOUND",
        "whisper_model": WHISPER_MODEL if STT_ENGINE == "whisper" else None,
        "whisper_ready": _whisper_model is not None,
        "vosk_models":   list(_vosk_models.keys()),
    }


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
