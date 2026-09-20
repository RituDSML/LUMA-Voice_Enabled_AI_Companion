# voice_output_ws.py — LUMA's Voice (Web version)
#
# Original voice_output.py called engine.say() / engine.runAndWait(), which
# plays audio through the SERVER's local speakers — in a deployed web app
# nobody is listening to the server, so this produces silence for the user.
#
# pyttsx3 can't return raw audio directly from say(), but it can render
# speech to a WAV file via save_to_file(). We use that to get audio bytes,
# then send those bytes to the browser over the WebSocket to play back.
# Same TTS engine and voice settings as the original — only the output
# destination changes (file -> bytes -> browser, instead of -> speakers).

import pyttsx3
import tempfile
import os
import re


# pyttsx3 has no concept of "skip this character" — its underlying engine
# (espeak on Linux, SAPI5 on Windows) reads emoji/symbols aloud by their
# Unicode name (e.g. "🌟" becomes the spoken word "glowing star"). This
# strips them from the text handed to the TTS engine ONLY — the original
# text (with emoji intact) is still what gets displayed in the chat UI,
# since main.py sends `text` and the synthesized audio as separate fields
# in the same message rather than deriving one from the other.
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols & pictographs, emoticons, transport, supplemental symbols
    "\U00002600-\U000026FF"  # misc symbols (☀ ☂ ⭐ etc.)
    "\U00002700-\U000027BF"  # dingbats (✂ ✈ ✉ etc.)
    "\U0001F1E6-\U0001F1FF"  # regional indicator symbols (flag letters)
    "\U00002B00-\U00002BFF"  # arrows, stars, misc symbols (⭐ ➡ etc.)
    "\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
    "\U00002300-\U000023FF"  # misc technical (⌚ ⏰ etc.)
    "\uFE0F"                 # variation selector (emoji presentation marker)
    "]+",
    flags=re.UNICODE,
)


def strip_for_speech(text: str) -> str:
    """
    Remove emoji and pictographic symbols before handing text to the TTS
    engine, so pyttsx3 doesn't verbalize their Unicode names. Collapses
    any resulting double-spaces left behind by the removal.
    """
    cleaned = _EMOJI_PATTERN.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def synthesize_speech(text: str) -> bytes:
    """
    Render text to speech using pyttsx3 and return the resulting WAV
    audio as bytes, ready to send to a client over a WebSocket.
    """
    engine = pyttsx3.init()

    voices = engine.getProperty("voices")
    if len(voices) > 1:
        engine.setProperty("voice", voices[1].id)
    # Lowered again per feedback: 175 -> 150 -> 135 wpm. This is a calmer,
    # more deliberate pace for a supportive companion voice.
    engine.setProperty("rate", 135)
    engine.setProperty("volume", 1.0)

    speech_text = strip_for_speech(text)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name

    try:
        engine.save_to_file(speech_text, temp_path)
        engine.runAndWait()
        engine.stop()

        with open(temp_path, "rb") as f:
            audio_bytes = f.read()
    finally:
        os.unlink(temp_path)

    return audio_bytes


# --- Quick manual check ---
if __name__ == "__main__":
    examples = [
        "That's wonderful to hear, Ritu! 🌟",
        "Great job! ⭐ Keep it up 💪",
        "No symbols here, just plain text.",
    ]
    for ex in examples:
        print(f"Original: {ex}")
        print(f"Spoken:   {strip_for_speech(ex)}\n")
