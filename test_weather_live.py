"""
test_weather_live.py — RUN THIS ON YOUR OWN MACHINE, not in a sandbox.
Verifies get_weather() against the real Open-Meteo API (no API key needed).

Usage:
    cd <your LUMA project folder>
    python test_weather_live.py

Covers: live verification of get_weather (Chapter 8, §8.3.3 placeholder item 1)
"""

from tools import get_weather

results = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail else ""))
    results.append(condition)
    return condition


print("=" * 60)
print("LIVE get_weather() TEST (real Open-Meteo API)")
print("=" * 60)

# --- Valid, well-known location ---
result = get_weather("Kochi")
print(f"\nQuery: 'Kochi'\nResult: {result}\n")
check("Valid location returns a temperature reading", "°C" in result and "currently" in result, result)
check("Valid location returns a wind speed reading", "km/h" in result, result)
check("Response does not contain an error message", "Sorry" not in result, result)

# --- Another valid location, to confirm it's not hardcoded/cached wrongly ---
result2 = get_weather("Wayanad")
print(f"Query: 'Wayanad'\nResult: {result2}\n")
check("Second valid location also returns a real reading", "°C" in result2, result2)
check("Different locations return different resolved place names",
      result.split(" in ")[1].split(",")[0] != result2.split(" in ")[1].split(",")[0]
      if "in " in result and "in " in result2 else True)

# --- Invalid / nonsense location ---
result3 = get_weather("Xyzzyplacethatdoesnotexist123")
print(f"Query: 'Xyzzyplacethatdoesnotexist123'\nResult: {result3}\n")
check("Invalid location returns a graceful 'not found' message, not a crash",
      "couldn't find" in result3.lower() or "sorry" in result3.lower(), result3)

print()
print("=" * 60)
print(f"{sum(results)}/{len(results)} checks passed")
print("=" * 60)
print("\nIf all checks passed, you can cite this in Chapter 8 §8.3.3 as live")
print("verification of get_weather against the real Open-Meteo API.")
