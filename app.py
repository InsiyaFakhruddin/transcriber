import streamlit as st
import os
import time
from pathlib import Path
from datetime import datetime
from tempfile import NamedTemporaryFile
import threading

# Assuming the meeting transcription and diarization logic is in `meeting_transcriber.py`
from meeting_transcriber import start_recording, stop_recording, transcribe_and_diarize, save_outputs, DEFAULT_MODEL, \
    DEFAULT_MIN_SPK, DEFAULT_MAX_SPK

# Streamlit page configuration
st.set_page_config(page_title="Meeting Transcriber", layout="centered")

# Create output directory if it doesn't exist
OUTPUT_DIR = Path.home() / "MeetingTranscripts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Helper function to display logs
def log(msg):
    st.text(msg)


# Helper function to start the background process for transcription
def transcribe_audio(wav_path, model_name, min_speakers, max_speakers):
    try:
        # Perform the transcription and diarization
        segments = transcribe_and_diarize(wav_path, model_name, min_speakers, max_speakers, log_cb=log)

        # Save the outputs
        md, srt, txt = save_outputs(wav_path, segments)

        # Display the saved file paths for the user
        st.success(f"Transcription and diarization complete! Files saved:\n- {md}\n- {srt}\n- {txt}")

        # Allow the user to download the transcript file (txt, md, srt)
        with open(txt, "r") as file:
            st.download_button("Download Transcript", file, file_name="transcript.txt", mime="text/plain")

        with open(md, "r") as file:
            st.download_button("Download Markdown", file, file_name="transcript.md", mime="text/markdown")

        with open(wav_path, "rb") as file:
            st.download_button("Download Audio", file, file_name="meeting_audio.wav", mime="audio/wav")
    except Exception as e:
        st.error(f"Error during transcription: {e}")


# Streamlit UI elements for user input
st.title("Meeting Transcriber (Web version)")
st.write("This tool records system audio and the mic, transcribes it, and diarizes the conversation.")

# Audio recording settings
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False
if "recording_state" not in st.session_state:
    st.session_state.recording_state = None

# Dropdown for model selection
model_choice = st.selectbox("Select Whisper Model", ["tiny.en", "base.en", "small.en", "medium"], index=1)
min_speakers = st.number_input("Min Speakers", min_value=1, max_value=10, value=DEFAULT_MIN_SPK)
max_speakers = st.number_input("Max Speakers", min_value=2, max_value=10, value=DEFAULT_MAX_SPK)

# Placeholder for status updates
status_text = st.empty()

# Audio file storage
wav_path = NamedTemporaryFile(delete=False, suffix=".wav").name

# Recording process
if "button_text" not in st.session_state:
    st.session_state.button_text = "Start Recording"  # Initial text for the button

# Toggle button based on recording state
start_stop_btn = st.button(st.session_state.button_text)

if start_stop_btn:
    if st.session_state.is_recording:
        # Stop the recording and initiate transcription
        status_text.text("Stopping recording...")
        stop_recording(st.session_state.recording_state)
        st.session_state.is_recording = False  # Reset the recording state
        status_text.text("Transcribing and diarizing...")

        # Show a progress bar while transcription is happening
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.03)  # Simulating transcription time
            progress_bar.progress(i + 1)

        # Once the transcription process is complete, proceed with file generation
        st.text("Transcript generated. Preparing download links...")
        transcribe_audio(st.session_state.recording_state.wav_path, model_choice, min_speakers, max_speakers)

        status_text.text("Process Complete!")
        st.session_state.button_text = "Start Recording"  # Change button text back to start
    else:
        # Start the recording and immediately change the button to "Stop Recording"
        st.session_state.is_recording = True
        st.session_state.button_text = "Stop Recording"  # Change button text to stop
        st.session_state.recording_state = start_recording()  # Record and store the state
        status_text.text(f"Recording started... Saving to: {st.session_state.recording_state.wav_path}")
        time.sleep(1)  # Short delay before button update
        st.session_state.button_text = "Stop Recording"  # Change button text to stop