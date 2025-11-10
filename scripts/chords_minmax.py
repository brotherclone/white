#!/usr/bin/env python3
"""Compute overall lowest/highest MIDI note values in a given directory (chords).
Prints a compact summary.
"""
import glob
import os
import sys

try:
    import mido
except Exception:
    print(
        "mido not installed; install with: python3 -m pip install mido", file=sys.stderr
    )
    raise

root = "/Volumes/LucidNonsense/White/chords"
files = sorted(glob.glob(os.path.join(root, "**", "*.mid"), recursive=True))
if not files:
    print("No .mid files found under", root)
    sys.exit(0)

global_min = 999
global_max = -1
min_files = []
max_files = []
file_count = 0
no_note_files = 0

for f in files:
    file_count += 1
    try:
        mid = mido.MidiFile(f)
    except Exception:
        # skip unreadable
        continue
    mn = None
    mx = None
    for track in mid.tracks:
        for msg in track:
            t = getattr(msg, "type", None)
            if t == "note_on":
                if getattr(msg, "velocity", 0) == 0:
                    continue
                n = getattr(msg, "note", None)
                if n is None:
                    continue
                mn = n if mn is None or n < mn else mn
                mx = n if mx is None or n > mx else mx
            elif t == "note_off":
                n = getattr(msg, "note", None)
                if n is None:
                    continue
                mn = n if mn is None or n < mn else mn
                mx = n if mx is None or n > mx else mx
    if mn is None and mx is None:
        no_note_files += 1
        continue
    if mn < global_min:
        global_min = mn
        min_files = [f]
    elif mn == global_min:
        min_files.append(f)
    if mx > global_max:
        global_max = mx
        max_files = [f]
    elif mx == global_max:
        max_files.append(f)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def name(n):
    return f"{NOTE_NAMES[n%12]}{(n//12)-1}"


print(f"Scanned {file_count} .mid files under: {root}")
print(f"Files with no note events: {no_note_files}")
if global_min <= global_max:
    print("\nOverall lowest MIDI note: {} -> {}".format(global_min, name(global_min)))
    print("Files containing lowest note (up to 10):")
    for p in min_files[:10]:
        print(" -", p)
    print("\nOverall highest MIDI note: {} -> {}".format(global_max, name(global_max)))
    print("Files containing highest note (up to 10):")
    for p in max_files[:10]:
        print(" -", p)
else:
    print("No note events found in any files under the directory")
