#!/usr/bin/env bash
# build.sh — Render build script for the VIET TTS+STT backend
# Runs once at deploy time. The resulting container image is what starts.
set -e   # exit immediately on any error

echo "=== [1/3] Installing Python dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== [2/3] Downloading Whisper model ==="
# XDG_CACHE_HOME is set in render.yaml to a path inside the project dir.
# This makes the downloaded model available to the running container.
python - <<'EOF'
import os, whisper

model_name = os.getenv("WHISPER_MODEL", "base")
print(f"  Downloading whisper/{model_name} to {os.getenv('XDG_CACHE_HOME', '~/.cache')} ...")
whisper.load_model(model_name)   # downloads if not cached, no-op if already there
print(f"  ✓ Model ready")
EOF

echo ""
echo "=== [3/3] Verifying imageio-ffmpeg ==="
python - <<'EOF'
import imageio_ffmpeg
exe = imageio_ffmpeg.get_ffmpeg_exe()
print(f"  ✓ ffmpeg binary: {exe}")
EOF

echo ""
echo "=== Build complete ==="
