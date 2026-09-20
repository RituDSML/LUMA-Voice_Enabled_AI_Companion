# main.py — LUMA FastAPI/WebSocket backend
#
# Single-user personal companion (no login/auth — multi-user registration
# is proposed as a future extension in the thesis rather than implemented
# here). Short-term history is session-scoped (in-memory, resets on
# refresh); long-term memory persists across sessions via memory.py.

import asyncio
import base64
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from emotion_model import get_emotion_label
from llm import get_response
from safety import check_crisis, CRISIS_RESPONSE
from database import (
    init_db,
    log_message,
    get_caregiver_info,
)
from memory import maybe_update_summary, get_current_summary
from reminders import start_scheduler, set_delivery_callback
from database import get_undelivered_reminders, mark_reminder_delivered
from voice_input_ws import transcribe_audio_bytes
from voice_output_ws import synthesize_speech
from graph import luma_graph

WELCOME_MESSAGE = (
    "Hello! I am LUMA, your personal health companion. "
    "I am here to listen and support you. How are you feeling today?"
)

app = FastAPI(title="LUMA")

# Single-user app: at most one WebSocket is ever meaningfully "active" at
# a time, so a module-level reference is enough to know whether to push
# a fired reminder live or leave it queued for catch-up on next connect.
_active_websocket = None
_main_event_loop = None


def _deliver_reminder(reminder_id, message, log_id):
    """
    Called from reminders.py's background scheduler thread — NOT the
    asyncio event loop, so we hand off to it via run_coroutine_threadsafe
    rather than awaiting directly here.
    """
    if _active_websocket is not None and _main_event_loop is not None:
        asyncio.run_coroutine_threadsafe(
            _send_reminder_live(message, log_id), _main_event_loop
        )
    # else: stays logged as undelivered, caught up on next connect (see
    # chat_socket below).


async def _send_reminder_live(message, log_id):
    try:
        await send_luma_message(_active_websocket, message)
        mark_reminder_delivered(log_id)
    except Exception as e:
        print(f"[reminders] Failed to deliver live reminder: {e}")


class ConversationState:
    def __init__(self):
        self.crisis_triggered = False
        self.crisis_spoken = False
        # Short-term (this-session-only) history, in memory. Starts empty
        # on every new connection/page refresh — NOT pulled from the
        # database, which holds every past session's messages. This is
        # what fixed the bug where a "fresh" chat after a refresh was
        # still silently getting old sessions' content (e.g. a past
        # playlist mention) fed into the LLM as context.
        self.history = []


async def send_json(ws: WebSocket, payload: dict):
    await ws.send_text(json.dumps(payload))


async def send_luma_message(ws: WebSocket, text: str, emotion: str | None = None,
                             include_audio: bool = True):
    """Log + send a LUMA response, with synthesized speech audio attached."""
    log_message("luma", text)
    audio_b64 = None
    if include_audio:
        try:
            audio_bytes = await asyncio.to_thread(synthesize_speech, text)
            audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        except Exception as e:
            print(f"[voice output error] {e}")
    await send_json(ws, {
        "type": "luma_message",
        "text": text,
        "emotion": emotion,
        "audio_b64": audio_b64,
    })


@app.on_event("startup")
async def startup():
    global _main_event_loop
    init_db()
    _main_event_loop = asyncio.get_running_loop()
    set_delivery_callback(_deliver_reminder)
    start_scheduler()


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.websocket("/ws/chat")
async def chat_socket(websocket: WebSocket):
    global _active_websocket
    await websocket.accept()
    _active_websocket = websocket
    state = ConversationState()

    log_message("luma", WELCOME_MESSAGE)
    history = [{"role": "luma", "content": WELCOME_MESSAGE, "emotion": None}]
    audio_bytes = await asyncio.to_thread(synthesize_speech, WELCOME_MESSAGE)
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    await send_json(websocket, {"type": "history", "messages": history})
    await send_json(websocket, {"type": "welcome_audio", "audio_b64": audio_b64})

    # Catch up on any reminders that fired while nobody was connected,
    # rather than letting them silently vanish.
    for entry in get_undelivered_reminders():
        await send_luma_message(websocket, entry["message"])
        mark_reminder_delivered(entry["id"])

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"] is not None:
                text = await asyncio.to_thread(transcribe_audio_bytes, message["bytes"])
                if text is None:
                    await send_json(websocket, {
                        "type": "transcription_failed",
                        "message": "Didn't catch that — try again.",
                    })
                    continue
                await send_json(websocket, {"type": "transcription", "text": text})
                await handle_user_text(websocket, state, text)

            elif "text" in message and message["text"] is not None:
                data = json.loads(message["text"])

                if data.get("type") == "user_message":
                    await handle_user_text(websocket, state, data["text"])

                elif data.get("type") == "resolve_crisis":
                    state.crisis_triggered = False
                    await send_json(websocket, {"type": "crisis_resolved"})

                elif data.get("type") == "request_caregiver":
                    info = get_caregiver_info()
                    await send_json(websocket, {"type": "caregiver_info", "info": info})

    except WebSocketDisconnect:
        pass
    finally:
        if _active_websocket is websocket:
            _active_websocket = None


async def handle_user_text(websocket: WebSocket, state: ConversationState, text: str):
    if state.crisis_triggered:
        return

    history = list(state.history)
    long_term_summary = get_current_summary()

    result = await asyncio.to_thread(luma_graph.invoke, {
        "user_input": text,
        "history": history,
        "long_term_summary": long_term_summary,
        "crisis_tier": "none",
        "emotion": None,
        "response": None,
    })

    if result["crisis_tier"] == "elevated":
        log_message("user", text, emotion=None)
        state.crisis_triggered = True
        state.crisis_spoken = True
        await send_json(websocket, {"type": "user_message", "text": text, "emotion": None})
        await send_luma_message(websocket, result["response"])
        await send_json(websocket, {"type": "crisis", "text": result["response"]})
        return

    emotion, response = result["emotion"], result["response"]
    log_message("user", text, emotion=emotion)
    await send_json(websocket, {"type": "user_message", "text": text, "emotion": emotion})
    await send_luma_message(websocket, response, emotion=None)

    state.history.append({"role": "user", "content": text})
    state.history.append({"role": "luma", "content": response})
    state.history = state.history[-12:]

    # Background long-term memory update — no-ops until threshold reached.
    asyncio.create_task(asyncio.to_thread(maybe_update_summary))


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
