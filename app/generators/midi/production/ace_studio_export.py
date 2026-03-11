#!/usr/bin/env python3
"""
ACE Studio Pipeline Export — Phase 3

Pushes an assembled melody and approved lyrics to the currently open ACE Studio
project via the MCP client.

Reads from a production directory:
  assembled/assembled_melody.mid  — assembled vocal melody MIDI
  melody/lyrics.txt               — approved lyric text (with # comments stripped)
  melody/review.yml               — singer name
  production_plan.yml             — BPM, time signature, title

The export:
  1. Configures the open project's BPM and time signature
  2. Loads the singer onto the first available track
  3. Adds a clip spanning the full assembled melody duration
  4. Opens the pattern editor and inserts all notes with the lyric sentence

Returns None (with a warning) if ACE Studio is unreachable or files are missing.

Usage:
    python -m app.generators.midi.production.ace_studio_export \
        --production-dir shrink_wrapped/.../production/green__last_pollinators_elegy_v1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import mido
import yaml

from app.reference.mcp.ace_studio.client import AceStudioClient

log = logging.getLogger(__name__)

# ACE Studio internal ticks-per-beat (standard for most DAWs and ACE Studio)
ACE_TPB = 480

# Default registry path (relative to project root via __file__)
_DEFAULT_REGISTRY = (
    Path(__file__).parents[4]
    / "app"
    / "reference"
    / "mcp"
    / "ace_studio"
    / "singer_voices.yml"
)


# ---------------------------------------------------------------------------
# Singer registry helpers
# ---------------------------------------------------------------------------


def load_singer_registry(path: Optional[Path] = None) -> dict:
    """Load singer_voices.yml and return a dict keyed by lowercase singer name.

    Returns {} (with a warning) if the file is missing or unreadable.
    """
    registry_path = Path(path) if path else _DEFAULT_REGISTRY
    if not registry_path.exists():
        log.warning("singer_voices.yml not found at %s; registry empty", registry_path)
        return {}
    try:
        data = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        singers = data.get("singers", {}) or {}
        return {k.lower(): v for k, v in singers.items()}
    except Exception as exc:
        log.warning("Failed to load singer registry: %s", exc)
        return {}


def resolve_ace_voice_name(singer_name: str, registry: dict) -> str:
    """Resolve a White project singer name to the ACE Studio voice name.

    Resolution order:
    1. Look up lowercase singer_name in registry.
    2. If found and ace_studio_voice is non-null, return that voice name.
    3. If found but ace_studio_voice is null, log warning and fall back to
       the White project name.
    4. If not in registry, log debug and fall back to the White project name.
    """
    if not singer_name:
        return singer_name

    entry = registry.get(singer_name.lower())

    if entry is None:
        log.debug(
            "Singer %r not in registry; passing name directly to find_singer()",
            singer_name,
        )
        return singer_name

    ace_voice = entry.get("ace_studio_voice")
    if ace_voice:
        return ace_voice

    log.warning(
        "Singer %r has no ace_studio_voice assigned in singer_voices.yml; "
        "falling back to White project name",
        singer_name,
    )
    return singer_name


# ---------------------------------------------------------------------------
# MIDI helpers
# ---------------------------------------------------------------------------


def parse_midi_notes(midi_path: Path) -> list[dict]:
    """Parse a MIDI file into a list of note dicts for add_notes_in_editor.

    Each dict: {"pos": int, "pitch": int, "dur": int} — all values in ACE_TPB ticks.
    Overlapping notes on the same pitch are handled by first-open-wins.
    """
    mid = mido.MidiFile(str(midi_path))
    tpb = mid.ticks_per_beat or ACE_TPB
    scale = ACE_TPB / tpb

    abs_tick = 0
    open_notes: dict[int, int] = {}  # pitch -> start_abs_tick
    notes: list[dict] = []

    # Assembled melody is a single merged track
    track = mid.tracks[0] if mid.tracks else []
    for msg in track:
        abs_tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            if msg.note not in open_notes:
                open_notes[msg.note] = abs_tick
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note in open_notes:
                start = open_notes.pop(msg.note)
                dur = abs_tick - start
                notes.append(
                    {
                        "pos": int(start * scale),
                        "pitch": msg.note,
                        "dur": max(1, int(dur * scale)),
                    }
                )

    return sorted(notes, key=lambda n: n["pos"])


def flatten_lyrics(lyrics_path: Path) -> str:
    """Flatten lyrics.txt to a space-joined lyric sentence for ACE Studio.

    Strips comment lines (# ...), section headers ([...]), instrumental
    markers, and blank lines. Remaining lines are the syllable/word content.
    """
    words: list[str] = []
    for line in lyrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            continue
        if line.startswith("[—") or line.startswith("[\u2014"):
            continue
        words.append(line)
    return " ".join(words)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_to_ace_studio(production_dir: str | Path) -> Optional[dict]:
    """Export assembled melody and lyrics to the currently open ACE Studio project.

    Args:
        production_dir: Path to the song production directory.

    Returns:
        dict with project_id, track_index, singer, note_count, title on success.
        None if ACE Studio is unreachable or required files are missing.
    """
    prod = Path(production_dir)

    assembled_midi = prod / "assembled" / "assembled_melody.mid"
    lyrics_file = prod / "melody" / "lyrics.txt"
    plan_file = prod / "production_plan.yml"
    melody_review_file = prod / "melody" / "review.yml"

    # Guard — required files
    for required in (assembled_midi, lyrics_file, plan_file):
        if not required.exists():
            log.warning("ACE Studio export skipped — missing: %s", required)
            return None

    # Load production plan
    plan = yaml.safe_load(plan_file.read_text(encoding="utf-8"))
    bpm = float(plan["bpm"])
    time_sig: str = plan.get("time_sig", "4/4")
    num, den = (int(x) for x in time_sig.split("/"))
    title: str = plan.get("title") or plan.get("song_slug", "")

    # Singer from melody review.yml (optional — export continues without singer)
    singer_name = ""
    if melody_review_file.exists():
        review = yaml.safe_load(melody_review_file.read_text(encoding="utf-8"))
        singer_name = review.get("singer", "")

    # Resolve White project name → ACE Studio voice name
    registry = load_singer_registry()
    ace_voice_name = resolve_ace_voice_name(singer_name, registry)

    # Parse inputs
    notes = parse_midi_notes(assembled_midi)
    if not notes:
        log.warning("ACE Studio export skipped — assembled MIDI contains no notes")
        return None

    total_ticks = max(n["pos"] + n["dur"] for n in notes)
    lyric_sentence = flatten_lyrics(lyrics_file)

    # Connect and export
    try:
        with AceStudioClient() as ace:
            # 1 — Track selection first (ACE Studio invalidates track list after
            # project metadata changes, so we must read tracks before set_tempo)
            tracks = ace.list_tracks()
            if not tracks:
                log.warning(
                    "ACE Studio export skipped — project has no tracks. "
                    "Add at least one track in ACE Studio before exporting."
                )
                return None
            track_index = 0

            # 2 — Project metadata (must come AFTER list_tracks; ACE Studio
            # returns an empty track list on subsequent calls after set_tempo)
            ace.set_tempo(bpm)
            ace.set_time_signature(num, den)

            # 3 — Singer
            singer_id: Optional[int] = None
            if ace_voice_name:
                candidates = ace.find_singer(ace_voice_name)
                if candidates:
                    singer_id = candidates[0].get("id")
                    if singer_id is not None:
                        ace.load_singer(track_index, singer_id)
                    else:
                        log.warning(
                            "Singer %r found but ID missing in response; skipping singer load",
                            ace_voice_name,
                        )
                else:
                    log.warning(
                        "Singer %r not found in ACE Studio sound sources; skipping singer load",
                        ace_voice_name,
                    )

            # 4 — Add clip spanning full assembled duration
            ace.add_clip(
                track_index=track_index,
                pos=0,
                dur=total_ticks,
                name=title or None,
            )

            # 5 — Open editor and insert notes + lyrics
            ace.open_editor()
            ace.add_notes_with_lyrics(notes, lyric_sentence)

            # 6 — Fetch project info for the return value
            project_info = ace.get_project_info()
            project_id: str = project_info.get("projectName") or title

            return {
                "project_id": project_id,
                "track_index": track_index,
                "singer": singer_name,
                "ace_studio_voice": ace_voice_name,
                "singer_id": singer_id,
                "note_count": len(notes),
                "total_ticks": total_ticks,
                "title": title,
            }

    except ConnectionError as exc:
        log.warning("ACE Studio export skipped — server unreachable: %s", exc)
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export assembled melody and lyrics to ACE Studio via MCP"
    )
    parser.add_argument(
        "--production-dir",
        required=True,
        help="Path to the song production directory",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    result = export_to_ace_studio(args.production_dir)
    if result is None:
        print("Export skipped — check warnings above.")
        sys.exit(1)

    print("Export complete:")
    print(f"  Project : {result['project_id']}")
    print(f"  Track   : {result['track_index']}")
    print(f"  Singer  : {result['singer'] or '(not loaded)'}")
    print(f"  Notes   : {result['note_count']}")
    sys.exit(0)


if __name__ == "__main__":
    main()
