#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Activate venv
source .venv/bin/activate

# Make sure pactl exists (comes from pulseaudio-utils)
if ! command -v pactl >/dev/null 2>&1; then
  echo "❌ pactl not found. Please run ./setup.sh first (installs pulseaudio-utils)."
  exit 1
fi

# Try to start a per-user PulseAudio daemon if none is running.
# (On PipeWire systems, pactl talks to pipewire-pulse and this is harmless.)
pulseaudio --check >/dev/null 2>&1 || pulseaudio --start --exit-idle-time=-1 || true

# Wait a moment for the server to come up
sleep 1

# Ensure we have a default sink; if not, create a virtual one with a monitor (rec.monitor)
DEFAULT_SINK="$(pactl get-default-sink 2>/dev/null || true)"
if [ -z "$DEFAULT_SINK" ]; then
  echo "ℹ️ No default sink; creating virtual sink 'rec'..."
  pactl load-module module-null-sink sink_name=rec sink_properties=device.description=rec >/dev/null
  pactl set-default-sink rec
  DEFAULT_SINK="rec"
fi

# Ensure we have at least one monitor source available
HAS_MONITOR="$(pactl list short sources | awk '{print $2}' | grep -c '\.monitor$' || true)"
if [ "$HAS_MONITOR" -eq 0 ]; then
  # If somehow no monitor is created, re-create the null sink to force a monitor
  pactl unload-module module-null-sink >/dev/null 2>&1 || true
  pactl load-module module-null-sink sink_name=rec sink_properties=device.description=rec >/dev/null
  pactl set-default-sink rec
fi

# Ensure a default mic source; if none, pick the first non-monitor source if available
DEFAULT_SOURCE="$(pactl get-default-source 2>/dev/null || true)"
if [ -z "$DEFAULT_SOURCE" ]; then
  FIRST_INPUT="$(pactl list short sources | awk '{print $2}' | grep -v '\.monitor$' | head -n1 || true)"
  if [ -n "$FIRST_INPUT" ]; then
    pactl set-default-source "$FIRST_INPUT"
    DEFAULT_SOURCE="$FIRST_INPUT"
  else
    echo "⚠️ No input (mic) source found. The app can still run, but mic won't record."
  fi
fi

echo "🔊 Audio ready:"
echo "  - Default sink:   $(pactl get-default-sink || echo '(none)')"
echo "  - Default source: $(pactl get-default-source || echo '(none)')"
echo "  - Available sources:"
pactl list short sources | awk '{print "    • " $2}'

# Launch Streamlit (0.0.0.0 so others on LAN can open it if needed)
exec streamlit run app.py --server.address 0.0.0.0
