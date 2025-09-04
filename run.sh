#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
pulseaudio --check || pulseaudio --start --exit-idle-time=-1
streamlit run app.py
