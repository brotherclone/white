#!/usr/bin/env python3
"""
Scan repository for .mid files and report highest and lowest MIDI note numbers per file.
Writes a JSON summary to midi_note_summary.json in the repo root and prints a compact table.
"""
import json
import os
import glob
import sys
from typing import Dict, Optional

try:
    import mido
except Exception:
    print(
        "The 'mido' package is required but not installed.\nInstall it with: python3 -m pip install mido",
        file=sys.stderr,
    )
    raise

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_number_to_name(n: int) -> str:
    octave = (n // 12) - 1
    name = NOTE_NAMES[n % 12]
    return f"{name}{octave}"


def analyze_file(path: str) -> Optional[Dict[str, object]]:
    try:
        mid = mido.MidiFile(path)
    except Exception as e:
        return {"path": path, "error": f"Failed to open MIDI: {e}"}

    min_note = None
    max_note = None

    for track in mid.tracks:
        for msg in track:
            # mido message types for notes
            if msg.type == "note_on":
                # note_on with velocity 0 is equivalent to note_off in many files
                if getattr(msg, "velocity", 0) == 0:
                    # treat as note_off; ignore
                    continue
                n = getattr(msg, "note", None)
                if n is None:
                    continue
                if min_note is None or n < min_note:
                    min_note = n
                if max_note is None or n > max_note:
                    max_note = n
            elif msg.type == "note_off":
                n = getattr(msg, "note", None)
                if n is None:
                    continue
                if min_note is None or n < min_note:
                    min_note = n
                if max_note is None or n > max_note:
                    max_note = n

    if min_note is None and max_note is None:
        return {
            "path": path,
            "min_note": None,
            "max_note": None,
            "min_name": None,
            "max_name": None,
        }

    return {
        "path": path,
        "min_note": min_note,
        "max_note": max_note,
        "min_name": note_number_to_name(min_note) if min_note is not None else None,
        "max_name": note_number_to_name(max_note) if max_note is not None else None,
    }


def find_mid_files(root: str = "."):
    patterns = ["**/*.mid"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(root, p), recursive=True))
    # exclude some build / egg-info / .git directories if present
    files = [
        f
        for f in files
        if "build" not in f and "egg-info" not in f and "__pycache__" not in f
    ]
    return sorted(files)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze MIDI files for min/max note numbers."
    )
    parser.add_argument(
        "--root", help="Root directory to search for .mid files (defaults to repo root)"
    )
    args = parser.parse_args()

    # determine repo root (one level up from scripts)
    default_repo_root = os.path.abspath(os.path.dirname(__file__))
    default_repo_root = os.path.dirname(default_repo_root)

    repo_root = args.root if args.root else default_repo_root
    # if a relative path was provided, make it relative to the repo root
    if not os.path.isabs(repo_root):
        repo_root = os.path.abspath(os.path.join(default_repo_root, repo_root))

    files = find_mid_files(repo_root)
    results = []

    for f in files:
        r = analyze_file(f)
        results.append(r)

    out_path = os.path.join(default_repo_root, "midi_note_summary.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)

    # Print concise table
    print(
        f"Analyzed {len(results)} .mid files under: {repo_root}\nSummary written to: {out_path}\n"
    )
    for r in results:
        if r is None:
            continue
        if "error" in r:
            print(f"ERROR: {r['path']}: {r['error']}")
            continue
        path = r["path"]
        mn = r.get("min_note")
        mx = r.get("max_note")
        mn_name = r.get("min_name")
        mx_name = r.get("max_name")
        if mn is None and mx is None:
            print(f"{path}: no note events found")
        else:
            print(f"{path}: lowest={mn} ({mn_name}), highest={mx} ({mx_name})")


if __name__ == "__main__":
    main()
