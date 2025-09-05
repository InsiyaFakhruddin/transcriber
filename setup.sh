#!/usr/bin/env bash
set -e

sudo apt update
sudo apt install -y \
  python3-venv python3-pip \
  ffmpeg \
  pulseaudio pulseaudio-utils pavucontrol \
  libsndfile1 \
  libgomp1 \
  python3-tk

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python -c "import streamlit" 2>/dev/null || pip install 'streamlit>=1.33,<2'

echo "✅ Setup complete."
echo "Next: ./run.sh"
