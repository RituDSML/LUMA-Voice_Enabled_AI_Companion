# mic_test.py — quick standalone check: does the microphone actually
# capture sound at the OS level, independent of the browser entirely?
#
# Records 4 seconds, saves to test_recording.wav, and reports basic
# volume stats so you can tell at a glance whether real audio was
# captured (vs silence).

import sounddevice as sd
import numpy as np
import wave

DURATION = 4  # seconds
SAMPLE_RATE = 16000

print("Recording will start in 1 second — say something after it starts...")
sd.sleep(1000)
print(f"Recording for {DURATION} seconds now — SPEAK NOW")

recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
sd.wait()
print("Recording finished.")

# Save it so you can play it back and listen yourself
with wave.open("test_recording.wav", "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)  # int16 = 2 bytes
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(recording.tobytes())

# Basic volume check — tells you immediately if it's just silence
max_amplitude = np.abs(recording).max()
avg_amplitude = np.abs(recording).mean()

print(f"\nMax amplitude: {max_amplitude} (out of 32767 possible)")
print(f"Average amplitude: {avg_amplitude:.1f}")

if max_amplitude < 500:
    print("\n⚠️  This looks like SILENCE — the mic may not be capturing sound,")
    print("    or the wrong input device is selected at the OS level.")
else:
    print("\n✅ Real audio was captured. Your microphone works fine at the OS level.")
    print("   Play test_recording.wav to confirm you can hear your voice.")

# List available input devices, in case the wrong one is default
print("\nAvailable input devices:")
print(sd.query_devices())
