#!/usr/bin/env python3
"""
ACE Studio MIDI Import — round-trip parser and LRC export.

Parses the vocal synthesis MIDI exported by ACE Studio (VocalSynthvN/VocalSynthvN_*.mid),
reconstructs word tokens from ACE's syllable-split lyric meta messages, and writes a
standard LRC subtitle file with word-level timestamps.

ACE Studio splits polysyllabic words into fragments: "iron" → "iron#1", "iron#2".
This module merges those fragments back into single word events.

Usage:
    python -m app.generators.midi.production.ace_studio_import \
        --production-dir shrink_wrapped/.../production/blue__rust_signal_memorial_v1

    python -m app.generators.midi.production.ace_studio_import \
        --production-dir ... --lrc-out path/to/output.lrc
"""

from __future__ import annotations

import argparse
import re
import sys

import mido

from pathlib import Path
from typing import Optional

PROPOSAL_FILENAME = "vocal_alignment.lrc"


# ---------------------------------------------------------------------------
# MIDI export discovery
# ---------------------------------------------------------------------------


def find_ace_export(production_dir: Path) -> Optional[Path]:
    """Locate the highest-versioned ACE Studio MIDI export in production_dir.

    Searches first for VocalSynthvN/VocalSynthvN_*.mid (versioned subfolder),
    then falls back to VocalSynthv*.mid directly in production_dir.
    Returns the MIDI from the highest-versioned match. Returns None if none found.
    """
    production_dir = Path(production_dir)

    # Primary: versioned subfolders VocalSynthvN/
    folders = sorted(production_dir.glob("VocalSynthv*/"), reverse=True)
    for folder in folders:
        if not folder.is_dir():
            continue
        midis = list(folder.glob("*.mid"))
        if midis:
            chosen = midis[0]
            if len(midis) > 1:
                import logging

                logging.getLogger(__name__).debug(
                    "Multiple MIDIs in %s, using %s", folder, chosen.name
                )
            return chosen

    # Fallback: VocalSynthv*.mid files directly in production_dir
    direct = sorted(production_dir.glob("VocalSynthv*.mid"), reverse=True)
    if direct:
        return direct[0]

    # Fallback: VocalSynthv*.mid inside melody/ subfolder
    melody_direct = sorted(
        (production_dir / "melody").glob("VocalSynthv*.mid"), reverse=True
    )
    if melody_direct:
        return melody_direct[0]

    return None


# ---------------------------------------------------------------------------
# MIDI parsing
# ---------------------------------------------------------------------------


def _parse_syllable_events(mid: mido.MidiFile) -> list[dict]:
    """Extract per-syllable events from a MIDI file.

    Each event dict:
        raw_text    — original lyric meta text (e.g. "iron#2")
        word        — base word without #N suffix (e.g. "iron")
        frag_index  — 1-based fragment index, or 0 for single-syllable words
        start_tick  — absolute tick of this syllable
        start_beat  — fractional beat (start_tick / ticks_per_beat)
        pitch       — MIDI note number at onset
        velocity    — velocity at onset
        end_tick    — absolute tick of note_off (0 if not resolved)
    """
    tpb = mid.ticks_per_beat or 480
    events: list[dict] = []

    for track in mid.tracks:
        tick = 0
        pending_lyric: Optional[dict] = None

        for msg in track:
            tick += msg.time

            if hasattr(msg, "text"):  # lyrics MetaMessage
                word_raw = msg.text
                m = re.match(r"^(.+?)#(\d+)$", word_raw)
                if m:
                    word, frag_index = m.group(1), int(m.group(2))
                else:
                    word, frag_index = word_raw, 0
                pending_lyric = {
                    "raw_text": word_raw,
                    "word": word,
                    "frag_index": frag_index,
                    "start_tick": tick,
                    "start_beat": tick / tpb,
                    "pitch": 0,
                    "velocity": 0,
                    "end_tick": 0,
                }
                events.append(pending_lyric)

            elif msg.type == "note_on" and msg.velocity > 0:
                if pending_lyric is not None and pending_lyric["pitch"] == 0:
                    pending_lyric["pitch"] = msg.note
                    pending_lyric["velocity"] = msg.velocity
                pending_lyric = None  # note consumed

            elif msg.type == "note_on" and msg.velocity == 0:
                # note_off: backfill end_tick on the most recent event with same pitch
                for ev in reversed(events):
                    if ev["pitch"] == msg.note and ev["end_tick"] == 0:
                        ev["end_tick"] = tick
                        break

    return events


# ---------------------------------------------------------------------------
# Syllable merging
# ---------------------------------------------------------------------------


def merge_syllables(syllable_events: list[dict]) -> list[dict]:
    """Collapse ACE syllable fragments into word-level events.

    "iron#1" + "iron#2" → {word: "iron", syllable_count: 2, start_beat: ..., end_beat: ...}
    Single-syllable words (frag_index == 0) pass through as syllable_count: 1.

    Returns list of word event dicts:
        word            — reconstructed word string
        syllable_count  — number of ACE fragments
        start_beat      — beat of the first fragment
        end_beat        — beat of the last fragment's end (or next event start)
        pitch           — MIDI note at onset
        velocity        — velocity at onset
        start_tick      — raw tick (for downstream use)
        end_tick        — raw tick of note end
    """
    if not syllable_events:
        return []

    words: list[dict] = []
    i = 0
    while i < len(syllable_events):
        ev = syllable_events[i]

        if ev["frag_index"] == 0:
            # Single-syllable word
            end_tick = ev["end_tick"] or (
                syllable_events[i + 1]["start_tick"]
                if i + 1 < len(syllable_events)
                else ev["start_tick"]
            )
            end_beat = (
                end_tick / (ev["start_tick"] / ev["start_beat"])
                if ev["start_beat"]
                else end_tick
            )
            # Recompute cleanly
            tpb_implied = (
                ev["start_tick"] / ev["start_beat"] if ev["start_beat"] else 480
            )
            end_beat = end_tick / tpb_implied if tpb_implied else ev["start_beat"]
            words.append(
                {
                    "word": ev["word"],
                    "syllable_count": 1,
                    "start_beat": ev["start_beat"],
                    "end_beat": end_beat,
                    "pitch": ev["pitch"],
                    "velocity": ev["velocity"],
                    "start_tick": ev["start_tick"],
                    "end_tick": end_tick,
                }
            )
            i += 1

        elif ev["frag_index"] == 1:
            # Collect all fragments for this word
            base_word = ev["word"]
            frags = [ev]
            j = i + 1
            while j < len(syllable_events):
                nxt = syllable_events[j]
                if nxt["word"] == base_word and nxt["frag_index"] == len(frags) + 1:
                    frags.append(nxt)
                    j += 1
                else:
                    break

            last = frags[-1]
            end_tick = last["end_tick"] or (
                syllable_events[j]["start_tick"]
                if j < len(syllable_events)
                else last["start_tick"]
            )
            tpb_implied = (
                ev["start_tick"] / ev["start_beat"] if ev["start_beat"] else 480
            )
            end_beat = end_tick / tpb_implied if tpb_implied else ev["start_beat"]
            words.append(
                {
                    "word": base_word,
                    "syllable_count": len(frags),
                    "start_beat": ev["start_beat"],
                    "end_beat": end_beat,
                    "pitch": ev["pitch"],
                    "velocity": ev["velocity"],
                    "start_tick": ev["start_tick"],
                    "end_tick": end_tick,
                }
            )
            i = j

        else:
            # Orphaned fragment (e.g. #2 without #1) — treat as single word
            words.append(
                {
                    "word": ev["word"],
                    "syllable_count": 1,
                    "start_beat": ev["start_beat"],
                    "end_beat": ev["start_beat"],
                    "pitch": ev["pitch"],
                    "velocity": ev["velocity"],
                    "start_tick": ev["start_tick"],
                    "end_tick": ev["end_tick"],
                }
            )
            i += 1

    return words


# ---------------------------------------------------------------------------
# LRC export
# ---------------------------------------------------------------------------


def _beat_to_timestamp(beat: float, tempo_us: int) -> str:
    """Convert a fractional beat to LRC timestamp [MM:SS.cc].

    tempo_us: microseconds per beat (1000000 = 60 BPM).
    """
    total_cs = round(
        beat * tempo_us / 10_000
    )  # integer centiseconds, no float overflow
    minutes, remainder = divmod(total_cs, 6000)
    secs, cs = divmod(remainder, 100)
    return f"[{minutes:02d}:{secs:02d}.{cs:02d}]"


def export_lrc(word_events: list[dict], tempo_us: int, output_path: Path) -> Path:
    """Write a standard LRC file from word events.

    Format: [MM:SS.cc] word  — one line per word, chronological order.
    """
    output_path = Path(output_path)
    lines = []
    for ev in word_events:
        ts = _beat_to_timestamp(ev["start_beat"], tempo_us)
        lines.append(f"{ts} {ev['word']}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_ace_export(midi_path: Path) -> tuple[list[dict], int]:
    """Parse an ACE Studio export MIDI.

    Returns (word_events, tempo_us) where word_events is the merged list
    and tempo_us is the MIDI tempo in microseconds per beat.
    """
    mid = mido.MidiFile(str(midi_path))

    # Extract tempo from track 0
    tempo_us = 500_000  # default 120 BPM
    for track in mid.tracks:
        for msg in track:
            if hasattr(msg, "tempo"):
                tempo_us = msg.tempo
                break

    syllables = _parse_syllable_events(mid)
    words = merge_syllables(syllables)
    return words, tempo_us


def load_ace_export(production_dir: Path) -> Optional[list[dict]]:
    """One-call helper: find the ACE export MIDI and return merged word events.

    Returns None (with a warning) if no export MIDI is found.
    """
    production_dir = Path(production_dir)
    midi_path = find_ace_export(production_dir)
    if midi_path is None:
        import logging

        logging.getLogger(__name__).warning(
            "No ACE Studio export MIDI found in %s", production_dir
        )
        return None
    words, _ = parse_ace_export(midi_path)
    return words


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse an ACE Studio vocal export MIDI and write an LRC file."
    )
    parser.add_argument("--production-dir", required=True)
    parser.add_argument(
        "--lrc-out",
        default=None,
        help="Output path for the LRC file (default: <production-dir>/vocal_alignment.lrc)",
    )
    args = parser.parse_args()

    prod = Path(args.production_dir)
    midi_path = find_ace_export(prod)
    if midi_path is None:
        print(f"ERROR: No ACE Studio export MIDI found in {prod}")
        sys.exit(1)

    print(f"Parsing: {midi_path.name}")
    word_events, tempo_us = parse_ace_export(midi_path)
    bpm = round(60_000_000 / tempo_us, 1)
    print(f"  Tempo: {bpm} BPM  |  Words: {len(word_events)}")

    lrc_path = Path(args.lrc_out) if args.lrc_out else prod / PROPOSAL_FILENAME
    export_lrc(word_events, tempo_us, lrc_path)
    print(f"  LRC written: {lrc_path}")


if __name__ == "__main__":
    main()
