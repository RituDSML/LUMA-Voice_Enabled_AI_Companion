# LUMA — Voice-Enabled, Emotion-Aware Conversational AI Companion

LUMA is a personal health companion combining mandatory crisis-safety
gating, concern-tier-gated retrieval-augmented grounding, and persistent
long-term memory in a single, auditable pipeline. This repository contains
the full source code and knowledge bases for the system evaluated in the
accompanying MSc thesis (Middlesex University Dubai, 2026).

**Thesis, chapter references, and full evaluation writeups are available
on request / in the submitted dissertation.** This README maps the files
in this repository to the chapters and sections that describe, implement,
and evaluate them.

## Setup

1. This repo does **not** include `.env`, `luma.db`, or `luma_emotion_model/`
   (secrets, real user data, and a large trained-model artifact
   respectively — see "What's excluded" below).
2. Create a `.env` file in the project root with:
   ```
   GROQ_API_KEY=your_key_here
   ```
3. Re-train the emotion classifier using the Colab notebook described in
   the Methodology chapter, and place the resulting `luma_emotion_model/`
   folder in the project root — or request the trained model directly.
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. `openai-whisper` needs `ffmpeg` on your system PATH.
6. `pyttsx3` needs a TTS engine available on the OS (SAPI5 on Windows,
   NSSpeechSynthesizer on macOS, espeak on Linux). Note: voice quality
   differs noticeably by platform, since pyttsx3 wraps whichever
   engine the OS provides rather than shipping its own — this is
   discussed as a known platform-dependent limitation in Chapter 9.

## Run

```
uvicorn main:app --reload
```

Then open **http://localhost:8000** in your browser and allow microphone
access when prompted.

## Repository layout → Thesis chapters

### Core application

| File | What it is | Thesis reference |
|---|---|---|
| `main.py` | FastAPI/WebSocket backend — the application entry point | Ch.5 §5.3 (architecture overview), Ch.6 §6.3 (implementation) |
| `graph.py` | LangGraph state graph — 5-node pipeline, crisis detection as entry point | Ch.5 §5.5–5.6 (design), Ch.6 §6.3 |
| `llm.py` | Response generation: concern-tier detection, gated RAG retrieval, bounded tool-calling loop | Ch.5 §5.6.4–5.6.5, Ch.6 §6.6–6.7 |
| `safety.py` | Keyword-based crisis detection (40 tracked phrases) | Ch.6 §6.5, Ch.7 §7.4 |
| `emotion_model.py` | Emotion classifier inference wrapper (BERT, 4-group schema) | Ch.6, Ch.7 §7.3 (training itself: Colab notebook, Methodology chapter) |
| `luma_rag.py` | FAISS + sentence-transformer RAG module over the two knowledge bases | Ch.5 §5.6.4, Ch.6 §6.7 |
| `database.py` | SQLite persistence — conversations, mood/habit logs, caregiver info, long-term memory, reminders | Ch.5 Table 5.5/5.5a, Ch.6 §6.9 |
| `memory.py` | Long-term memory summarization (allow-list/deny-list scoped) | Ch.6 §6.8 |
| `reminders.py` | Background scheduler for daily recurring reminders | Ch.6, Ch.7 (reminder firing tests) |
| `tools.py` | Native tool-calling functions (`get_weather`, `log_habit`, `schedule_reminder`, etc.) | Ch.6 §6.6 |
| `voice_input_ws.py` | Speech-to-text via Whisper (browser audio → transcription) | Ch.6 §6.11 |
| `voice_output_ws.py` | Text-to-speech via pyttsx3 (text → browser-playable audio) | Ch.6 §6.11 |
| `static/` | Frontend (chat UI, crisis-response view, voice controls) | Ch.6 §6.3, §6.11 |

### Knowledge bases

| File | What it is | Thesis reference |
|---|---|---|
| `coping_strategies.json` | 10 curated CBT/DBT-sourced coping techniques | Ch.5 §5.6.4, Ch.7 §7.5 |
| `first_aid.json` | 8 curated first-aid/emergency entries | Ch.5 §5.6.4, Ch.7 §7.5 |

### Crisis detection — cosine-similarity upgrade

| File | What it is | Thesis reference |
|---|---|---|
| `test_crisis_detection.py` | 42-utterance labelled test set (14 direct / 14 paraphrased / 14 negative control) | Ch.7 §7.4.1–7.4.2 |
| `crisis_detection_cosine.py` | Embedding-based cosine-similarity detector + hybrid (substring OR cosine) function | Ch.7 §7.4.4 |
| `evaluate_crisis_cosine.py` | Threshold sweep comparing substring / cosine / hybrid methods | Ch.7 §7.4.4 |
| `crisis_cosine_threshold_sweep.csv`, `crisis_cosine_similarity_detail.csv` | Full results output from the sweep above | Ch.7 §7.4.4, Appendix |

### RAG quantitative evaluation

| File | What it is | Thesis reference |
|---|---|---|
| `rag_eval_questions.json` | 30 grounded questions (12 first-aid, 18 coping; half paraphrased) | Ch.7 §7.5.1 |
| `rag_eval.py` | Hit-rate@k evaluation script | Ch.7 §7.5.3 |
| `rag_eval_results.csv`, `rag_eval_summary.md` | Hit-rate@k results | Ch.7 §7.5.2–7.5.3 |
| `generate_answers_for_faithfulness.py` | Generates answers for faithfulness scoring | Ch.7 §7.5.4 |
| `rag_faithfulness_scoring.csv` | Manually-scored faithfulness results (Faithful/Partial/Unfaithful) | Ch.7 §7.5.4 |

### Automated testing

| File | What it is | Thesis reference |
|---|---|---|
| `evaluate_test_set.py` | Emotion classifier test-set evaluation | Ch.7 §7.3 |
| `test_tools.py`, `test_tool_chaining.py` | Tool-calling unit tests, incl. `MAX_TOOL_ROUNDS` safety-cap verification | Ch.7 §7.5/8.3.3 |
| `test_reminder_firing.py` | Reminder scheduling/firing/catch-up-delivery tests | Ch.7 §7.5/8.3.3 |
| `test_database.py` | Persistence-across-restart tests | Ch.7 §7.5/8.3.3 |
| `test_memory_live.py` | Long-term memory allow-list/deny-list live test (hallucination-fix verification) | Ch.8 §8.3.3 |
| `test_caregiver.py` | Caregiver data-layer tests (save/replace/persist-across-restart, 8/8 passing) | Ch.6 §6.5.1 |
| `test_weather_live.py` | `get_weather` live API test (incl. the Wayanad geocoding limitation) | Ch.6, Ch.9 (limitations) |
| `test_llm_wiring.py` | LLM response-generation wiring tests | Ch.7 |
| `mic_test.py` | Manual microphone diagnostic utility (not a formal automated test) | — |

### Deployment

| File | What it is | Thesis reference |
|---|---|---|
| `Dockerfile`, `docker-compose.yml`, `Dockerfile.test` | Containerization, evaluated but not used for the primary demo | Ch.9 (deployment discussion — native execution used due to a platform-dependent pyttsx3/TTS-engine limitation in the Linux container) |
| `requirements.txt`, `requirements-test.txt` | Python dependencies | Setup |

## What's excluded from this repository

- **`.env`** — contains the Groq API key. Never committed; see Setup above.
- **`luma.db`** — the real SQLite database, containing actual conversation
  history, mood/habit logs, and caregiver information. Excluded to be
  consistent with the data-handling position discussed in Chapter 8: this
  data is stored locally and not intended for public distribution.
- **`luma_emotion_model/`** — the trained BERT emotion classifier (~400MB).
  Too large for a plain git repository and not source code; retrain via
  the Colab notebook described in the Methodology chapter.

## Known limitations

See Chapter 9 (Limitations and Future Work) for the full discussion.
Briefly: crisis detection's substring-matching baseline has strong recall
on direct statements but weak recall on paraphrased distress (Ch.7 §7.4);
an embedding-based hybrid upgrade is implemented here (`crisis_detection_cosine.py`)
and evaluated in §7.4.4. The RAG concern-tier gate similarly under-fires
on natural phrasing (Ch.7 §7.5.2). The system is single-user, with no
authentication, by deliberate scope decision (Ch.1 §1.2).
