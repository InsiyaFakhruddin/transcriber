#!/usr/bin/env bash
set -e

# System deps (includes pactl via pulseaudio-utils, libsndfile1 for soundfile, ffmpeg for recording)
sudo apt update
sudo apt install -y \
  python3-venv python3-pip \
  ffmpeg \
  pulseaudio pulseaudio-utils pavucontrol \
  libsndfile1 \
  libgomp1

# Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# Python deps
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Safety: if streamlit wasn't in requirements for some reason
python -c "import streamlit" 2>/dev/null || pip install 'streamlit>=1.33,<2'

echo "✅ Setup complete."
echo "Next: ./run.sh  (this will start audio + launch the app)"
