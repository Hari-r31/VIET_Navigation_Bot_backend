# ── Stage 1: Build — download Whisper model at build time ─────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download Whisper model into /app/.cache/whisper
ARG WHISPER_MODEL=base
ENV XDG_CACHE_HOME=/app/.cache
RUN python -c "\
import whisper, os; \
m = os.environ.get('WHISPER_MODEL', 'base'); \
print(f'Downloading whisper/{m} ...'); \
whisper.load_model(m, download_root='/app/.cache/whisper'); \
print('Done')"

# ── Stage 2: Runtime — lean final image ───────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy pre-downloaded model cache
COPY --from=builder /app/.cache /app/.cache

# Copy application code
COPY . .

# Whisper reads model from this cache dir
ENV XDG_CACHE_HOME=/app/.cache
ENV STT_ENGINE=whisper
ENV WHISPER_MODEL=base

# Render / VPS: listen on $PORT (default 5000)
ENV PORT=5000
EXPOSE 5000

CMD ["sh", "-c", "uvicorn tts_server:app --host 0.0.0.0 --port ${PORT}"]
