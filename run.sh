cp run.sh run.sh.bak 2>/dev/null || true
cat > run.sh <<'EOF'
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Activate venv
source .venv/bin/activate

# Need pactl (pulseaudio-utils)
if ! command -v pactl >/dev/null 2>&1; then
  echo "❌ pactl not found. Run ./setup.sh first."
  exit 1
fi

# Start per-user PulseAudio if not running
pulseaudio --check >/dev/null 2>&1 || pulseaudio --start --exit-idle-time=-1 || true
sleep 1

# Ensure default sink; if missing, create a virtual sink with monitor
DEFAULT_SINK="$(pactl get-default-sink 2>/dev/null || true)"
if [ -z "$DEFAULT_SINK" ]; then
  echo "ℹ️ No default sink; creating virtual sink 'rec'..."
  pactl load-module module-null-sink sink_name=rec sink_properties=device.description=rec >/dev/null
  pactl set-default-sink rec
  DEFAULT_SINK="rec"
fi

# Ensure at least one *.monitor source exists
HAS_MONITOR="$(pactl list short sources | awk '{print $2}' | grep -c '\.monitor$' || true)"
if [ "$HAS_MONITOR" -eq 0 ]; then
  pactl unload-module module-null-sink >/dev/null 2>&1 || true
  pactl load-module module-null-sink sink_name=rec sink_properties=device.description=rec >/dev/null
  pactl set-default-sink rec
fi

# Ensure default mic source; if none, pick first non-monitor source
DEFAULT_SOURCE="$(pactl get-default-source 2>/dev/null || true)"
if [ -z "$DEFAULT_SOURCE" ]; then
  FIRST_INPUT="$(pactl list short sources | awk '{print $2}' | grep -v '\.monitor$' | head -n1 || true)"
  if [ -n "$FIRST_INPUT" ]; then
    pactl set-default-source "$FIRST_INPUT"
    DEFAULT_SOURCE="$FIRST_INPUT"
  else
    echo "⚠️ No input (mic) source found. App runs but mic won't record."
  fi
fi

echo "🔊 Audio ready:"
echo "  - Default sink:   $(pactl get-default-sink || echo '(none)')"
echo "  - Default source: $(pactl get-default-source || echo '(none)')"
echo "  - Sources:"
pactl list short sources | awk '{print "    • " $2}'

# Launch Streamlit (bind all interfaces)
exec streamlit run app.py --server.address 0.0.0.0
EOF
chmod +x run.sh
