#!/usr/bin/env python3
"""
ACE Studio Drift Report — compare vocal synthesis export against approved loops.

Parses the ACE Studio MIDI export, segments word events by arrangement section,
and compares each section's actual pitch/rhythm against the approved melody loop.
Writes drift_report.yml with per-section and overall metrics.

Usage:
    python -m app.generators.midi.production.drift_report \
        --production-dir shrinkwrapped/.../production/blue__rust_signal_memorial_v1
"""

from __future__ import annotations

import argparse
import sys
import yaml
import mido

from pathlib import Path
from typing import Optional

from app.generators.midi.production.ace_studio_import import (
    find_ace_export,
    parse_ace_export,
)

DRIFT_REPORT_FILENAME = "drift_report.yml"
MELODY_TRACK = 4  # Logic Pro track number for melody in arrangement.txt


# ---------------------------------------------------------------------------
# Timecode parsing
# ---------------------------------------------------------------------------


def _parse_timecode(tc: str) -> float:
    """Parse a Logic Pro SMPTE timecode to seconds from song start.

    Format: HH:MM:SS:FF.sf  (30fps, 100 subframes per frame)
    Song start offset is 01:00:00:00.00, so HH=1 → 0 seconds.
    """
    parts = tc.strip().split(":")
    hh, mm, ss = int(parts[0]), int(parts[1]), int(parts[2])
    frame_parts = parts[3].split(".")
    ff = int(frame_parts[0])
    sf = int(frame_parts[1]) if len(frame_parts) > 1 else 0
    return (hh - 1) * 3600 + mm * 60 + ss + ff / 30.0 + sf / 3000.0


# ---------------------------------------------------------------------------
# Arrangement parsing
# ---------------------------------------------------------------------------


def _parse_arrangement_sections(
    arrangement_path: Path, track: int = MELODY_TRACK
) -> list[tuple[str, float, float]]:
    """Return (label, start_sec, end_sec) for the given track number.

    arrangement.txt format: start_tc  label  track_num  end_tc
    """
    sections: list[tuple[str, float, float]] = []
    with open(arrangement_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            start_tc, label, track_num, end_tc = (
                parts[0],
                parts[1],
                int(parts[2]),
                parts[3],
            )
            if track_num != track:
                continue
            sections.append((label, _parse_timecode(start_tc), _parse_timecode(end_tc)))
    return sections


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def segment_ace_export_by_arrangement(
    word_events: list[dict], arrangement_path: Path, bpm: int
) -> dict[str, list]:
    """Map ACE word events to arrangement sections.

    Uses track 4 (melody) clips from arrangement.txt as section boundaries.
    Words before the first section, or in gaps between sections, are omitted.
    Words straddling a boundary are assigned to the section where they start.

    Returns {section_label: [word_event, ...]} — empty sections omitted.
    """
    sections = _parse_arrangement_sections(arrangement_path, track=MELODY_TRACK)
    result: dict[str, list] = {label: [] for label, _, _ in sections}

    for word in word_events:
        word_sec = word["start_beat"] * 60.0 / bpm
        for label, start_sec, end_sec in sections:
            if start_sec <= word_sec < end_sec:
                result[label].append(word)
                break

    return {label: words for label, words in result.items() if words}


# ---------------------------------------------------------------------------
# Approved MIDI loading
# ---------------------------------------------------------------------------


def _load_approved_notes(midi_path: Path) -> list[dict]:
    """Extract note-on events from an approved loop MIDI.

    Returns list of {pitch, start_beat} sorted by onset.
    """
    try:
        mid = mido.MidiFile(str(midi_path))
    except Exception:
        return []
    tpb = mid.ticks_per_beat or 480
    notes: list[dict] = []
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                notes.append({"pitch": msg.note, "start_beat": tick / tpb})
    return sorted(notes, key=lambda n: n["start_beat"])


# ---------------------------------------------------------------------------
# Levenshtein distance (word-level)
# ---------------------------------------------------------------------------


def _levenshtein(a: list[str], b: list[str]) -> int:
    """Edit distance between two word sequences."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


# ---------------------------------------------------------------------------
# Section comparison
# ---------------------------------------------------------------------------


def compare_section(approved_midi_path: Path, ace_word_events: list[dict]) -> dict:
    """Compare approved melody loop against ACE word events for one section.

    Returns:
        pitch_match_pct     — fraction of note pairs within 2 semitones (by position)
        rhythm_drift_beats  — mean absolute onset difference (approved vs ACE, beat units)
        lyric_edit_distance — None (computed globally by generate_drift_report)
        note_count_delta    — ACE note count minus approved note count
    """
    approved = _load_approved_notes(approved_midi_path)
    ace = sorted(ace_word_events, key=lambda w: w["start_beat"])

    n_approved = len(approved)
    n_ace = len(ace)
    n_pairs = min(n_approved, n_ace)

    pitch_match_pct: Optional[float] = None
    rhythm_drift_beats: Optional[float] = None

    if n_pairs > 0:
        # Section start beat — align ACE onsets to loop-relative time
        section_start_beat = ace[0]["start_beat"]

        pitch_matches = sum(
            1
            for i in range(n_pairs)
            if abs(approved[i]["pitch"] - ace[i]["pitch"]) <= 2
        )
        pitch_match_pct = round(pitch_matches / n_pairs, 4)

        drifts = [
            abs(approved[i]["start_beat"] - (ace[i]["start_beat"] - section_start_beat))
            for i in range(n_pairs)
        ]
        rhythm_drift_beats = round(sum(drifts) / len(drifts), 4)

    return {
        "pitch_match_pct": pitch_match_pct,
        "rhythm_drift_beats": rhythm_drift_beats,
        "lyric_edit_distance": None,
        "note_count_delta": n_ace - n_approved if n_approved > 0 else None,
    }


# ---------------------------------------------------------------------------
# Global lyric edit distance
# ---------------------------------------------------------------------------


def _compute_lyric_edits(production_dir: Path, ace_words: list[dict]) -> Optional[int]:
    """Levenshtein distance between approved lyrics and ACE word sequence.

    Tries melody/lyrics.txt first, then melody/lyrics_draft.txt.
    Returns None if no lyrics file found.
    """
    melody_dir = production_dir / "melody"
    for path in [melody_dir / "lyrics.txt", melody_dir / "lyrics_draft.txt"]:
        if path.exists():
            text = path.read_text(encoding="utf-8")
            approved_words = [
                w
                for line in text.splitlines()
                for w in line.lower().split()
                if not line.strip().startswith("#") and not line.strip().startswith("[")
            ]
            ace_word_list = [w["word"].lower() for w in ace_words]
            return _levenshtein(approved_words, ace_word_list)
    return None


# ---------------------------------------------------------------------------
# BPM / time_sig loader
# ---------------------------------------------------------------------------


def _load_bpm(production_dir: Path) -> int:
    """Read BPM from chords/review.yml, default 120."""
    review_path = production_dir / "chords" / "review.yml"
    if review_path.exists():
        with open(review_path) as f:
            data = yaml.safe_load(f) or {}
        return int(data.get("bpm", 120))
    return 120


def _load_time_sig(production_dir: Path) -> str:
    """Read time_sig from chords/review.yml, default 4/4."""
    review_path = production_dir / "chords" / "review.yml"
    if review_path.exists():
        with open(review_path) as f:
            data = yaml.safe_load(f) or {}
        return str(data.get("time_sig", "4/4"))
    return "4/4"


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------


def generate_drift_report(production_dir: Path) -> dict:
    """Load ACE export, segment by arrangement, compare each section.

    Returns a dict with overall metrics and per-section breakdown.
    Raises FileNotFoundError if ACE export or arrangement.txt is missing.
    """
    production_dir = Path(production_dir)

    # Load ACE export
    midi_path = find_ace_export(production_dir)
    if midi_path is None:
        raise FileNotFoundError(f"No ACE Studio export MIDI found in {production_dir}")
    word_events, _ = parse_ace_export(midi_path)

    # Load arrangement
    arrangement_path = production_dir / "arrangement.txt"
    if not arrangement_path.exists():
        raise FileNotFoundError(f"arrangement.txt not found in {production_dir}")

    bpm = _load_bpm(production_dir)
    melody_approved_dir = production_dir / "melody" / "approved"

    # Segment words into sections
    segmented = segment_ace_export_by_arrangement(word_events, arrangement_path, bpm)

    # Per-section comparison
    sections_data: list[dict] = []
    pitch_matches: list[float] = []
    rhythm_drifts: list[float] = []

    for label, words in segmented.items():
        midi_path = melody_approved_dir / f"{label}.mid"
        if midi_path.exists():
            result = compare_section(midi_path, words)
        else:
            result = {
                "pitch_match_pct": None,
                "rhythm_drift_beats": None,
                "lyric_edit_distance": None,
                "note_count_delta": None,
            }

        result["section"] = label
        result["word_count"] = len(words)
        sections_data.append(result)

        if result["pitch_match_pct"] is not None:
            pitch_matches.append(result["pitch_match_pct"])
        if result["rhythm_drift_beats"] is not None:
            rhythm_drifts.append(result["rhythm_drift_beats"])

    overall_pitch_match = (
        round(sum(pitch_matches) / len(pitch_matches), 4) if pitch_matches else None
    )
    overall_rhythm_drift = (
        round(sum(rhythm_drifts) / len(rhythm_drifts), 4) if rhythm_drifts else None
    )
    total_lyric_edits = _compute_lyric_edits(production_dir, word_events)

    return {
        "overall_pitch_match": overall_pitch_match,
        "overall_rhythm_drift": overall_rhythm_drift,
        "total_lyric_edits": total_lyric_edits,
        "total_word_count": len(word_events),
        "sections": sections_data,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_drift_report(production_dir: Path, report: dict) -> Path:
    """Write drift_report.yml to production_dir."""
    out_path = Path(production_dir) / DRIFT_REPORT_FILENAME
    with open(out_path, "w") as f:
        yaml.dump(
            report, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate drift report comparing ACE Studio export against approved loops."
    )
    parser.add_argument("--production-dir", required=True)
    args = parser.parse_args()

    prod = Path(args.production_dir)
    if not prod.exists():
        print(f"ERROR: Production directory not found: {prod}")
        sys.exit(1)

    print(f"Generating drift report for: {prod.name}")
    try:
        report = generate_drift_report(prod)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    out_path = write_drift_report(prod, report)

    n_sections = len(report["sections"])
    print(f"  Sections compared: {n_sections}")
    print(f"  Total words: {report['total_word_count']}")
    if report["overall_pitch_match"] is not None:
        print(f"  Overall pitch match: {report['overall_pitch_match']:.1%}")
    if report["overall_rhythm_drift"] is not None:
        print(f"  Overall rhythm drift: {report['overall_rhythm_drift']:.3f} beats")
    if report["total_lyric_edits"] is not None:
        print(f"  Lyric edits: {report['total_lyric_edits']}")
    print(f"  Written: {out_path}")


if __name__ == "__main__":
    main()
