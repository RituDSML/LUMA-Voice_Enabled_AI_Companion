# LUMA — FastAPI/WebSocket backend with voice I/O, RAG, and LangGraph
#
# Assumes a FLAT project layout: main.py and every other .py module sit
# in the same directory (no src/ or app/ subfolder). Adjust the COPY
# paths below if your real layout differs.

FROM python:3.11-slim

# --- System packages ---------------------------------------------------
# ffmpeg    : Whisper shells out to it to decode .webm/.ogg audio from
#             the browser's MediaRecorder
# espeak-ng : pyttsx3's TTS backend on Linux (no SAPI5 here, unlike
#             Windows where you've been running it locally)
# build-essential : some pip packages (e.g. faiss-cpu deps) need a
#             compiler if no prebuilt wheel matches this base image
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    espeak-ng \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies ------------------------------------------------
# Copied and installed before the rest of the code so Docker can cache
# this layer — it won't reinstall ~2GB of torch/whisper on every code
# change, only when requirements.txt itself changes.
COPY requirements.txt .

# Install CPU-only torch FIRST, from PyTorch's own index — this skips
# the default GPU build, which otherwise pulls several GB of unneeded
# NVIDIA/CUDA packages (triton, nvidia-cublas, nvidia-cudnn, etc). Your
# machine has no NVIDIA GPU and LUMA never uses one, so this is purely
# waste we're cutting out — same functionality, much smaller/faster.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# --- Pre-download models at BUILD time, not first request ---------------
# Without this, the container's first user would wait through a slow
# on-demand download (and it would fail entirely on a host with
# restricted egress). Baking them in makes startup fast and offline-safe.
RUN python -c "import whisper; whisper.load_model('base')"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# --- App code -------------------------------------------------------------
# Includes: main.py, emotion_model.py, llm.py, safety.py, database.py,
# memory.py, reminders.py, voice_input_ws.py, voice_output_ws.py,
# graph.py, tools.py, luma_rag.py, coping_strategies.json,
# first_aid.json, static/, and luma_emotion_model/ (your fine-tuned
# BERT weights — this is the one thing that makes the image large;
# that's expected and fine).
COPY . .

EXPOSE 8000

# NOTE: main.py must be lowercase exactly "main.py" — Linux filesystems
# are case-sensitive, unlike Windows where "Main.py"/"main.py" collide.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]