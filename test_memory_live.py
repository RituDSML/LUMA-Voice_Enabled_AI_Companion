"""
test_memory_live.py — RUN THIS ON YOUR OWN MACHINE, not in a sandbox.
Needs a real GROQ_API_KEY in your .env (this makes a real LLM call).

Seeds a controlled, synthetic conversation into an ISOLATED test database
(does not touch your real luma.db), then forces memory.py's summarizer to
run, and prints the resulting summary for you to inspect against the
checklist below.

The conversation deliberately includes:
  - Facts that SHOULD appear in the summary (name, age, job, a health
    reading, a daily check-in fact)
  - Content that must NEVER appear (emotional state, a one-off book
    recommendation, crisis-adjacent language)
  - A "trap" scenario shaped like the original hospital-visit bug: LUMA
    asks a question, the person gives an ambiguous/non-confirming answer,
    and the summary must NOT record it as a confirmed fact.

Usage:
    cd <your LUMA project folder>
    python test_memory_live.py
"""

import os
import sys
import tempfile

TEST_DB = tempfile.mktemp(suffix="_luma_memtest.db")

import database
database.DB_PATH = TEST_DB
database.init_db()

import memory

# --- Seed a synthetic conversation ---
conversation = [
    ("user", "Hi, I'm Ritz, I'm 32 and I work as a data scientist.", "neutral"),
    ("luma", "Nice to meet you, Ritz! What are you working on?", None),
    ("user", "Studying for my MSc in Data Science at Middlesex University.", "neutral"),
    ("luma", "That sounds great. How are you feeling about it?", None),
    ("user", "Honestly pretty stressed and overwhelmed with the deadline.", "negative"),
    ("luma", "That sounds tough. Did you go to the hospital about your headache yesterday?", None),
    ("user", "I'm not sure, maybe later this week.", "ambiguous"),  # <-- TRAP: must NOT become "went to hospital"
    ("user", "My blood pressure this morning was 120/80.", "neutral"),
    ("luma", "Good to know. Anything else today?", None),
    ("user", "I took my medication this morning and went for a walk.", "neutral"),
    ("user", "Also you should read Atomic Habits, it's a great book.", "neutral"),
]

for role, message, emotion in conversation:
    database.log_message(role, message, emotion)

print(f"Seeded {len(conversation)} messages into isolated test DB: {TEST_DB}")
print(f"SUMMARY_UPDATE_THRESHOLD = {memory.SUMMARY_UPDATE_THRESHOLD}")
print(f"Messages logged: {len(conversation)} "
      f"({'>=' if len(conversation) >= memory.SUMMARY_UPDATE_THRESHOLD else '<'} threshold)")

if len(conversation) < memory.SUMMARY_UPDATE_THRESHOLD:
    print("\nNot enough messages to trigger summarization — lower "
          "memory.SUMMARY_UPDATE_THRESHOLD temporarily or add more turns above.")
    sys.exit(1)

print("\nCalling maybe_update_summary() — this makes a REAL Groq API call...\n")
memory.maybe_update_summary()

summary = memory.get_current_summary()

print("=" * 60)
print("RESULTING SUMMARY:")
print("=" * 60)
print(summary or "(empty)")
print("=" * 60)

print("""
MANUALLY CHECK THE SUMMARY ABOVE AGAINST THIS CHECKLIST:

SHOULD be present:
  [ ] Name: Ritz
  [ ] Age: 32
  [ ] Occupation: data scientist
  [ ] Education: MSc Data Science, Middlesex University
  [ ] Blood pressure reading: 120/80
  [ ] Daily check-in: took medication, went for a walk

MUST NOT be present (this is the actual hallucination-fix test):
  [ ] Any mention of "hospital" or a hospital visit having occurred
      -> the person only said "not sure, maybe later this week" —
         if the summary states or implies a hospital visit HAPPENED,
         the fix has NOT held and this is a genuine, reproducible
         recurrence of the original bug.
  [ ] Emotional state ("stressed", "overwhelmed") or any interpretation
      of mood/feelings
  [ ] The Atomic Habits book recommendation (one-off, out of scope)
  [ ] Any crisis/self-harm-adjacent language (none was in this test
      conversation, but confirm none was invented)

If every unchecked "must not" box stays empty and every "should" box is
present, the fix holds under this test case. If the hospital line leaks
through, that's a genuine, reproducible finding worth reporting exactly
as such in Chapter 8 rather than assuming the fix is complete.
""")

os.remove(TEST_DB)
