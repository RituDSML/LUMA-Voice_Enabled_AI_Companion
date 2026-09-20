# memory.py — LUMA's long-term memory
#
# Short-term memory (last few turns) already works via database.py's
# get_recent_conversations(), passed as `history` into get_response().
# That covers "what did we just say" but not "what do I know about this
# person" across sessions/days — this module adds that second layer.
#
# Approach: every SUMMARY_UPDATE_THRESHOLD new messages, take the messages
# since the last summary + the existing summary itself, and ask the LLM to
# produce one updated, compact summary. That summary is what gets injected
# into LUMA's system prompt on every turn (see llm.py).

from groq import Groq
from dotenv import load_dotenv
import os

from database import (
    get_long_term_summary,
    save_long_term_summary,
    get_conversations_after,
    get_latest_message_id,
)

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# How many new messages accumulate before we bother re-summarizing.
# Lower = more up-to-date memory but more LLM calls; higher = cheaper but
# memory lags behind more. 10 is a reasonable starting point.
SUMMARY_UPDATE_THRESHOLD = 10

SUMMARY_SYSTEM_PROMPT = """You maintain a compact long-term memory summary for LUMA, \
an AI companion, about the person it talks to. You will be given the EXISTING \
summary (may be empty) and a batch of NEW conversation turns. Produce an UPDATED \
summary that merges both.

STRICT SCOPE — only include these categories, nothing else:
- Name
- Age (only if explicitly stated)
- Job / occupation
- Schooling / education (degree, course, institution — only if explicitly stated)
- Health readings the person explicitly reports (e.g. blood pressure, blood \
sugar/glucose readings, or similar vitals)
- Any ailment or medical condition the person explicitly states they have
- Daily check-in facts, ONLY if explicitly stated as having happened (not asked \
about): what time they woke up, whether they took their medication, whether \
they did their exercise/routine check for the day. Keep these as dated facts \
(e.g. "Jul 20: woke 7am, took BP medication, did not exercise") rather than \
vague generalizations, so old daily entries don't get treated as still true today.

Do NOT include habit-tracking details (smoking, drinking, etc.) here — those are \
already tracked separately in a dedicated log and don't belong in this summary.

STRICT EXCLUSIONS — never include any of the following, even if it seems relevant \
or was discussed at length:
- Emotional state, mood, mental health topics, grief, relationship difficulties, \
stress, or any interpretation of how they're feeling
- Anything not explicitly and plainly stated by the person as fact — do not \
infer, extrapolate, or connect dots (e.g. "in a hurry" does NOT mean a medical \
emergency; a hypothetical question LUMA itself asked is NOT a confirmed fact, \
even if the person didn't clearly deny it)
- One-off events, single mentions, or anything that only matters for the current \
conversation, not future ones
- Anything related to crisis, self-harm, or safety escalation — that is handled \
entirely separately and must never appear here
- Book/media recommendations, one-time requests, or small talk

If nothing in the new turns fits the strict scope above, do not add anything — \
it is completely fine for the summary to stay short or even empty.
For dated daily check-in entries specifically: keep only the most recent 5 days' \
worth: when merging, drop older dated entries beyond that so the summary doesn't \
grow indefinitely.
Keep the whole summary under 120 words, plain bullet points, no interpretation.
Respond with ONLY the updated summary text — no preamble, no explanation."""


def _format_turns_for_summary(rows):
    lines = []
    for r in rows:
        speaker = "Person" if r["role"] == "user" else "LUMA"
        lines.append(f"{speaker}: {r['message']}")
    return "\n".join(lines)


def maybe_update_summary():
    """
    Check whether enough new messages have accumulated since the last
    summary update; if so, regenerate the summary and save it. Call this
    after logging each turn (see main.py). Safe/cheap to call often — it
    no-ops if the threshold hasn't been reached.
    """
    existing_summary, last_id = get_long_term_summary()
    latest_id = get_latest_message_id()

    new_message_count = latest_id - last_id
    if new_message_count < SUMMARY_UPDATE_THRESHOLD:
        return  # not enough new material yet, skip

    new_rows = get_conversations_after(last_id)
    if not new_rows:
        return

    new_turns_text = _format_turns_for_summary(new_rows)
    existing_text = existing_summary or "(no existing summary yet)"

    user_prompt = f"EXISTING SUMMARY:\n{existing_text}\n\nNEW CONVERSATION TURNS:\n{new_turns_text}"

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        updated_summary = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[memory] Summary update failed, keeping old summary: {e}")
        return

    save_long_term_summary(updated_summary, latest_id)


def get_current_summary():
    """Fetch the current long-term summary text (or None if none/empty)."""
    summary, _ = get_long_term_summary()
    return summary if summary else None


def reset_summary():
    """
    Wipe the stored long-term summary completely (e.g. after discovering
    hallucinated/incorrect content in it, like the false hospital-visit
    entry). Does NOT touch short-term history (recent_conversations) —
    only the persistent cross-session summary.
    """
    save_long_term_summary("", get_latest_message_id())
    print("[memory] Long-term summary cleared.")


# Test
if __name__ == "__main__":
    from database import init_db
    init_db()
    print("Current summary:", get_current_summary())
    maybe_update_summary()
    print("After update attempt:", get_current_summary())
