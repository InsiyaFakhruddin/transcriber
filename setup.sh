#!/usr/bin/env bash
set -e
sudo apt update
sudo apt install -y python3-venv python3-pip ffmpeg pulseaudio pavucontrol libsndfile1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
echo "✅ Setup complete. Next: source .venv/bin/activate && streamlit run app.py"
