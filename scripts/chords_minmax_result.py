#!/usr/bin/env python3
import glob
import os
import json
import sys

try:
    import mido
except Exception:
    print("mido not installed", file=sys.stderr)
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
no_note = 0
for f in files:
    try:
        mid = mido.MidiFile(f)
    except Exception:
        continue
    mn = None
    mx = None
    for tr in mid.tracks:
        for msg in tr:
            t = getattr(msg, "type", None)
            if t == "note_on":
                if getattr(msg, "velocity", 0) == 0:
                    continue
                n = getattr(msg, "note", None)
            elif t == "note_off":
                n = getattr(msg, "note", None)
            else:
                n = None
            if n is None:
                continue
            if mn is None or n < mn:
                mn = n
            if mx is None or n > mx:
                mx = n
    if mn is None and mx is None:
        no_note += 1
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

res = {
    "root": root,
    "file_count": len(files),
    "files_with_no_note_events": no_note,
    "global_min": global_min,
    "global_min_note_name": None,
    "global_min_files": min_files,
    "global_max": global_max,
    "global_max_note_name": None,
    "global_max_files": max_files,
}
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
if global_min <= global_max:
    res["global_min_note_name"] = f"{NOTE_NAMES[global_min%12]}{(global_min//12)-1}"
    res["global_max_note_name"] = f"{NOTE_NAMES[global_max%12]}{(global_max//12)-1}"
out_path = os.path.join(os.path.dirname(root), "chords_minmax_result.json")
with open(out_path, "w") as fh:
    json.dump(res, fh, indent=2)
print("WROTE", out_path)
