#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meeting Scribe (CPU-only): Start/Stop GUI to record system audio + mic, then transcribe + diarize.
Outputs: .md, .srt, .txt saved in ~/MeetingTranscripts

- Recording: FFmpeg via PulseAudio/PipeWire monitor + default mic (no Zoom/Meet banner).
- ASR: faster-whisper (CPU int8) with word_timestamps=True.
- Diarization: ECAPA embeddings on sliding windows + Agglomerative (cosine) with silhouette model selection.
- Robust: energy gate (dBFS) + smoothed 0.1s speaker track → per-word labels → one block per speaker turn, with short interjections kept.

Legal: Ensure recording/transcription complies with laws and policies where you use this.
"""

import platform

IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"


import signal
import queue
import threading
import subprocess
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch

from faster_whisper import WhisperModel
from speechbrain.inference import EncoderClassifier
from scipy.signal import medfilt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

import tkinter as tk
from tkinter import ttk, messagebox

# ---------------------- Config ----------------------
DEFAULT_MODEL = "base.en"      # tiny.en / base.en / small.en / medium (CPU)
EMB_WIN = 2.0                  # seconds (embedding window) — longer = stabler speakers
EMB_HOP = 0.5                  # seconds (hop) — finer resolution of changes
TRACK_STEP = 0.10              # seconds (speaker track resolution)
SMOOTH_KERNEL = 7              # median filter kernel (odd)
MIN_HOLD_S = 0.9               # min duration to keep a run as a turn (reduce if over-merge)
MAX_INTERJECT_S = 0.7          # keep short different-speaker runs as interjections
RMS_THRESH_DBFS = -48.0        # energy gate for windows (skip very quiet segments)
DEFAULT_MIN_SPK = 2            # expected minimum speakers
DEFAULT_MAX_SPK = 6            # maximum speakers

OUTPUT_DIR = Path.home() / "MeetingTranscripts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------- Utilities ----------------------
def run_cmd(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()

def get_default_sources():
    """Return (monitor_source_name, mic_source_name) using pactl (PulseAudio/PipeWire)."""
    rc, default_sink, _ = run_cmd(["pactl", "get-default-sink"])
    if rc != 0 or not default_sink:
        raise RuntimeError("Cannot get default sink. Is PulseAudio/PipeWire running?")

    rc, out, _ = run_cmd(["pactl", "list", "short", "sources"])
    monitor = None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            name = parts[1]
            if name.endswith(".monitor") and default_sink in name:
                monitor = name
                break
    if not monitor:
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].endswith(".monitor"):
                monitor = parts[1]
                break
    if not monitor:
        raise RuntimeError("Could not locate monitor source (system audio).")

    rc, default_source, _ = run_cmd(["pactl", "get-default-source"])
    if rc != 0 or not default_source:
        raise RuntimeError("Cannot get default source (mic).")

    return monitor, default_source

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------------------- Recording ----------------------
@dataclass
class RecordingState:
    process: subprocess.Popen | None = None
    wav_path: Path | None = None
    running: bool = False

import platform
import subprocess
from pathlib import Path
import signal
from datetime import datetime

# ---------------------------
# OS detection
IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"
# ---------------------------

OUTPUT_DIR = Path.home() / "MeetingTranscripts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

class RecordingState:
    def __init__(self, process, wav_path, running=True):
        self.process = process
        self.wav_path = wav_path
        self.running = running

def start_recording(sample_rate=16000):
    out = OUTPUT_DIR / f"meeting_{timestamp()}.wav"

    if IS_MAC:
        # macOS: AVFoundation, select audio device
        # Replace '1' with your actual microphone ID from `ffmpeg -f avfoundation -list_devices true -i ""`
        audio_device_id = "1"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-f", "avfoundation", "-i", f":{audio_device_id}",
            "-ac", "1", "-ar", str(sample_rate),
            "-c:a", "pcm_s16le", str(out)
        ]

    elif IS_LINUX:
        # Linux: PulseAudio
        mon, mic = get_default_sources()  # your existing function
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-f", "pulse", "-i", mon,
            "-f", "pulse", "-i", mic,
            "-filter_complex", "amix=inputs=2:duration=longest:dropout_transition=3,aresample=16000,pan=mono|c0=0.5*c0+0.5*c1",
            "-ac", "1", "-ar", str(sample_rate),
            "-c:a", "pcm_s16le", str(out)
        ]

    else:
        raise RuntimeError(f"Unsupported OS: {platform.system()}")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return RecordingState(process=proc, wav_path=out, running=True)

def stop_recording(state: RecordingState, wait_timeout=5):
    if not state or not state.process or not state.running:
        return
    try:
        state.process.send_signal(signal.SIGINT)   # clean stop
    except Exception:
        pass
    try:
        state.process.wait(timeout=wait_timeout)
    except subprocess.TimeoutExpired:
        state.process.kill()
    state.running = False


# ---------------------- Audio helpers ----------------------
def load_audio_mono16k(path):
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    return wav, sr

def seconds_to_srt(ts):
    h = int(ts // 3600); m = int((ts % 3600) // 60); s = int(ts % 60)
    ms = int((ts - int(ts)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def save_outputs(base_path: Path, segments):
    """
    segments: list of dict {spk, start, end, text}
    """
    base_no_ext = base_path.parent / base_path.stem

    md_path = base_no_ext.with_suffix(".md")
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Meeting Transcript — {base_no_ext.name}\n\n")
        for seg in segments:
            f.write(f"**SPEAKER_{seg['spk']}** [{seg['start']:.1f}–{seg['end']:.1f}]: {seg['text']}\n\n")

    txt_path = base_no_ext.with_suffix(".txt")
    with txt_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"SPEAKER_{seg['spk']} [{seg['start']:.1f}-{seg['end']:.1f}]: {seg['text']}\n")

    srt_path = base_no_ext.with_suffix(".srt")
    with srt_path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{seconds_to_srt(seg['start'])} --> {seconds_to_srt(seg['end'])}\n")
            f.write(f"SPEAKER_{seg['spk']}: {seg['text']}\n\n")

    return md_path, srt_path, txt_path

# ---------------------- Embeddings & Clustering ----------------------
def window_iter(audio, sr, win_s, hop_s):
    n = len(audio); w = int(sr * win_s); h = int(sr * hop_s)
    if w <= 0 or h <= 0:
        return
    for start in range(0, max(1, n - w + 1), h):
        st_s = start / sr
        en_s = (start + w) / sr
        yield st_s, en_s, audio[start:start+w]

def dbfs(x: np.ndarray) -> float:
    # x expected float in [-1, 1]
    rms = np.sqrt(np.mean(np.square(x))) + 1e-12
    return 20.0 * np.log10(rms)

def compute_embeddings(audio, sr, log=lambda *_: None):
    """
    Sliding ECAPA embeddings with energy gating.
    Returns: windows ([(st,en), ...]), embs (np.ndarray), ok (bool)
    """
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"}
    )

    windows, embs = [], []
    for st, en, seg in window_iter(audio, sr, EMB_WIN, EMB_HOP):
        if len(seg) < int(0.2*sr):
            continue
        if dbfs(seg) < RMS_THRESH_DBFS:
            continue
        xt = torch.from_numpy(seg).float().unsqueeze(0)  # [1, T]
        with torch.no_grad():
            rep = classifier.encode_batch(xt).squeeze().cpu().numpy()
        embs.append(rep); windows.append((st, en))

    if not embs:
        log("No voiced/energetic windows detected for diarization.")
        return [], np.zeros((0,)), False

    embs = normalize(np.vstack(embs))  # cosine-friendly
    return windows, embs, True

def choose_k_and_cluster(embs, min_k, max_k, log=lambda *_: None):
    """
    Choose K by maximizing silhouette score (cosine).
    Always enforces k >= min_k (if enough samples).
    """
    n = len(embs)
    if n < 2:
        return np.zeros(n, dtype=int)

    usable_max = min(max_k, n)  # cannot exceed samples
    best_k, best_score, best_labels = None, -1.0, None

    for k in range(max(2, min_k), max(3, usable_max + 1)):
        try:
            model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
            labels = model.fit_predict(embs)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(embs, labels, metric="cosine")
            if score > best_score:
                best_k, best_score, best_labels = k, score, labels
        except Exception:
            continue

    if best_labels is None:
        # Fallback: force min_k (or 1 if not enough samples)
        k = min_k if n >= min_k else 1
        log(f"Silhouette selection inconclusive; forcing k={k}.")
        if k == 1:
            return np.zeros(n, dtype=int)
        model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        return model.fit_predict(embs)

    log(f"Selected k={best_k} by silhouette (score={best_score:.3f}).")
    return best_labels

# ---------------------- Main pipeline ----------------------
def transcribe_and_diarize(wav_path: Path, model_name, min_speakers, max_speakers, log_cb=None):
    log = (lambda msg: log_cb(msg)) if log_cb else print

    log("Loading audio...")
    audio, sr = load_audio_mono16k(str(wav_path))
    if sr != 16000:
        tmp = wav_path.with_suffix(".tmp.wav")
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", str(wav_path), "-ac", "1", "-ar", "16000",
                        "-c:a", "pcm_s16le", str(tmp)], check=True)
        audio, sr = load_audio_mono16k(str(tmp))
        try:
            tmp.unlink()
        except Exception:
            pass

    log(f"ASR (faster-whisper {model_name}, word timestamps)...")
    asr = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _ = asr.transcribe(str(wav_path),
                                 vad_filter=True,
                                 vad_parameters={"min_silence_duration_ms": 300},
                                 word_timestamps=True)

    words = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                if w.word:
                    words.append({"start": float(w.start), "end": float(w.end), "word": w.word.strip()})
        else:
            words.append({"start": float(seg.start), "end": float(seg.end), "word": seg.text.strip()})

    if not words:
        log("No words from ASR; returning empty transcript.")
        return []

    log("Computing sliding-window embeddings...")
    win_list, embs, ok = compute_embeddings(audio, sr, log=log)
    if not ok or len(win_list) == 0:
        log("Diarization unavailable; using single-speaker transcript.")
        return merge_words_into_turns(words)

    log("Clustering embeddings (auto-K with silhouette)...")
    labels = choose_k_and_cluster(embs, min_speakers, max_speakers, log=log)

    # Build dense speaker track over time, smooth, and enforce min-hold
    dur = max(words[-1]["end"], len(audio)/sr)
    t_grid = np.arange(0.0, dur + 1e-9, TRACK_STEP)

    def label_for_t(t):
        # choose window whose center is closest, preferring those that cover t
        best, best_idx = -1e9, -1
        for i, (st, en) in enumerate(win_list):
            c = 0.5 * (st + en)
            score = -abs(c - t) + (2.0 if st <= t <= en else 0.0)
            if score > best:
                best, best_idx = score, i
        return int(labels[best_idx]) if (best_idx != -1 and len(labels)) else 0

    raw_track = np.array([label_for_t(t) for t in t_grid], dtype=int)
    smooth_track = medfilt(raw_track, kernel_size=SMOOTH_KERNEL if SMOOTH_KERNEL % 2 else SMOOTH_KERNEL + 1)

    min_hold_frames = max(1, int(round(MIN_HOLD_S / TRACK_STEP)))
    max_intr_frames = max(1, int(round(MAX_INTERJECT_S / TRACK_STEP)))

    # compress runs
    runs = []
    s_idx = 0
    for i in range(1, len(smooth_track) + 1):
        if i == len(smooth_track) or smooth_track[i] != smooth_track[i - 1]:
            runs.append([smooth_track[s_idx], s_idx, i - 1])  # [label, start_frame, end_frame]
            s_idx = i

    # merge runs shorter than MIN_HOLD unless they are small interjections
    i = 0
    while i < len(runs):
        lab, s, e = runs[i]
        length = e - s + 1
        if length < min_hold_frames and length > max_intr_frames:
            left = runs[i - 1] if i - 1 >= 0 else None
            right = runs[i + 1] if i + 1 < len(runs) else None
            if left and left[0] == lab and right and right[0] == lab:
                left[2] = right[2]; del runs[i:i + 2]; i -= 1; continue
            elif left and left[0] == lab:
                left[2] = e; del runs[i]; i -= 1; continue
            elif right and right[0] == lab:
                right[1] = s; del runs[i]; continue
        i += 1

    final_track = np.zeros_like(smooth_track)
    for lab, s, e in runs:
        final_track[s:e + 1] = lab

    def label_from_track(t):
        idx = int(round(t / TRACK_STEP))
        idx = max(0, min(len(final_track) - 1, idx))
        return int(final_track[idx])

    # Assign per-word speakers from final track
    for w in words:
        mid = 0.5 * (w["start"] + w["end"])
        w["spk"] = label_from_track(mid)

    # Merge into turns (only break when speaker changes)
    return merge_words_into_turns(words)

def merge_words_into_turns(words):
    segments_out = []
    cur = None
    for w in words:
        spk = w.get("spk", 0)
        if cur is None:
            cur = {"spk": spk, "start": w["start"], "end": w["end"], "text": w["word"]}
            continue
        if spk == cur["spk"]:
            joiner = "" if (w["word"].startswith("'") or cur["text"].endswith("'")) else " "
            cur["end"] = w["end"]
            cur["text"] += joiner + w["word"]
        else:
            segments_out.append(cur)
            cur = {"spk": spk, "start": w["start"], "end": w["end"], "text": w["word"]}
    if cur:
        segments_out.append(cur)
    return segments_out

# ---------------------- GUI ----------------------
class MeetingScribeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Meeting Scribe (CPU)")
        self.root.geometry("740x430")
        self.state = None
        self.worker = None
        self.log_queue = queue.Queue()
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.min_spk_var = tk.IntVar(value=DEFAULT_MIN_SPK)
        self.max_spk_var = tk.IntVar(value=DEFAULT_MAX_SPK)

        frm = ttk.Frame(root, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        # Model
        ttk.Label(frm, text="Whisper model (CPU):").grid(row=0, column=0, sticky="w")
        self.model_dd = ttk.Combobox(frm, textvariable=self.model_var,
                                     values=["tiny.en", "base.en", "small.en", "medium"],
                                     state="readonly", width=12)
        self.model_dd.grid(row=0, column=1, sticky="w", padx=(8, 16))

        # Min / Max speakers
        ttk.Label(frm, text="Min speakers:").grid(row=0, column=2, sticky="e")
        self.min_spk_dd = ttk.Combobox(frm, textvariable=self.min_spk_var,
                                       values=[1,2,3,4,5,6,7,8], state="readonly", width=4)
        self.min_spk_dd.grid(row=0, column=3, sticky="w", padx=(6, 16))

        ttk.Label(frm, text="Max speakers:").grid(row=0, column=4, sticky="e")
        self.max_spk_dd = ttk.Combobox(frm, textvariable=self.max_spk_var,
                                       values=[2,3,4,5,6,7,8,9,10], state="readonly", width=4)
        self.max_spk_dd.grid(row=0, column=5, sticky="w")

        self.start_btn = ttk.Button(frm, text="Start Recording", command=self.on_start, width=20)
        self.stop_btn  = ttk.Button(frm, text="Stop & Transcribe", command=self.on_stop, width=20, state="disabled")
        self.start_btn.grid(row=1, column=0, pady=10, sticky="w")
        self.stop_btn.grid(row=1, column=1, pady=10, sticky="w")

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(frm, textvariable=self.status_var).grid(row=2, column=0, columnspan=6, sticky="w")

        self.log_txt = tk.Text(frm, height=16, wrap="word")
        self.log_txt.grid(row=3, column=0, columnspan=6, sticky="nsew", pady=(8, 0))
        for c in range(6):
            frm.columnconfigure(c, weight=1 if c in (1,3,5) else 0)
        frm.rowconfigure(3, weight=1)

        self.root.after(200, self._poll_logs)

    def log(self, msg):
        self.log_queue.put(msg)

    def _poll_logs(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_txt.insert("end", msg + "\n")
                self.log_txt.see("end")
        except queue.Empty:
            pass
        self.root.after(200, self._poll_logs)

    def on_start(self):
        try:
            self.status_var.set("Starting recording...")
            self.state = start_recording()
            self.status_var.set(f"Recording → {self.state.wav_path}")
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.log(f"[{datetime.now().strftime('%H:%M:%S')}] Recording started.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start recording:\n{e}")
            self.status_var.set("Ready.")

    def on_stop(self):
        if not self.state or not self.state.running:
            return
        self.status_var.set("Stopping recording...")
        stop_recording(self.state)
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.log(f"[{datetime.now().strftime('%H:%M:%S')}] Recording stopped.")
        wav_path = self.state.wav_path
        self.status_var.set("Transcribing + diarizing (CPU)...")

        args = (wav_path, self.model_var.get(), int(self.min_spk_var.get()), int(self.max_spk_var.get()))
        self.worker = threading.Thread(target=self._process_file, args=args, daemon=True)
        self.worker.start()

    def _process_file(self, wav_path: Path, model_name: str, min_spk: int, max_spk: int):
        try:
            def cb(m): self.log(m)
            # ensure min <= max
            min_spk = max(1, min(min_spk, max_spk))
            max_spk = max(min_spk, max_spk)
            segments_out = transcribe_and_diarize(wav_path, model_name, min_spk, max_spk, log_cb=cb)
            md, srt, txt = save_outputs(wav_path, segments_out)
            self.log(f"Saved:\n- {md}\n- {srt}\n- {txt}")
            self.status_var.set("Done. Files saved in ~/MeetingTranscripts")
        except Exception as e:
            self.status_var.set("Failed.")
            self.log(f"ERROR: {e}")
            messagebox.showerror("Error", f"Processing failed:\n{e}")

def main():
    root = tk.Tk()
    app = MeetingScribeApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()