cp setup.sh setup.sh.bak 2>/dev/null || true
cat > setup.sh <<'EOF'
#!/usr/bin/env bash
set -e

# System deps (pactl via pulseaudio-utils, ffmpeg, Tk, etc.)
sudo apt update
sudo apt install -y \
  python3-venv python3-pip \
  ffmpeg \
  pulseaudio pulseaudio-utils pavucontrol \
  libsndfile1 \
  libgomp1 \
  python3-tk

# Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# Python deps
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Safety: ensure Streamlit present
python -c "import streamlit" 2>/dev/null || pip install 'streamlit>=1.33,<2'

echo "✅ Setup complete."
echo "Next: ./run.sh  (starts audio + launches the app)"
EOF
chmod +x setup.sh
