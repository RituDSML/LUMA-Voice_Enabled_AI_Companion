# voice_input_ws.py — LUMA's Ears (Web version)
#
# Original voice_input.py recorded directly from the server's local
# microphone via sounddevice — that only works when the app and the mic
# are on the same machine (i.e. your terminal app).
#
# In the web version, the BROWSER captures audio (via MediaRecorder) and
# sends it to the server as bytes over the WebSocket. This module takes
# those bytes and runs them through the same Whisper model you already
# fine-tuned the pipeline around — the transcription step itself is
# unchanged, only how the audio arrives is different.

import whisper
import tempfile
import os

_model = None


def get_model():
    """Load Whisper once, on first use (same lazy-load pattern as original)."""
    global _model
    if _model is None:
        _model = whisper.load_model("base")
    return _model


def transcribe_audio_bytes(audio_bytes: bytes, suffix: str = ".webm") -> str | None:
    """
    Transcribe raw audio bytes (as received from the browser's MediaRecorder)
    using Whisper. Returns the transcribed text, or None if nothing usable
    was captured.

    suffix should match what the browser sends (MediaRecorder commonly
    produces .webm or .ogg — whisper/ffmpeg handles both transparently,
    unlike the original which wrote .wav from sounddevice directly).
    """
    print(f"[voice_input_ws] Received {len(audio_bytes) if audio_bytes else 0} bytes")  # ADDED

    if not audio_bytes:
        print("[voice_input_ws] No bytes received — check frontend/WebSocket")  # ADDED
        return None

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        temp_path = f.name
        f.write(audio_bytes)

    try:
        result = get_model().transcribe(temp_path, language="en", fp16=False)
        text = result["text"].strip()
    finally:
        os.unlink(temp_path)

    return text if text else None