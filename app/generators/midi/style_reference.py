"""MIDI Style Reference — feature extraction and profile caching.

Extracts statistical features from MIDI files for `sounds_like` artists,
caches them as YAML profiles, and aggregates across artists into a single
StyleProfile written to `song_context.yml`.

Local directory layout (relative to project root or configurable):
    style_references/
        grouper/
            dragging_a_dead_deer.mid
            profile.yml          ← cached profile
        beach_house/
            space_song.mid
            profile.yml

Usage:
    from app.generators.midi.style_reference import (
        load_or_extract_profile,
        aggregate_profiles,
    )

    profiles = [load_or_extract_profile(a, refs_dir) for a in sounds_like]
    agg = aggregate_profiles([p for p in profiles if p is not None])
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import mido
import yaml

from app.structures.music.style_profile import StyleProfile

log = logging.getLogger(__name__)

# Default style_references directory (alongside the project root)
DEFAULT_STYLE_REFS_DIR = Path(__file__).parents[4] / "style_references"


# ---------------------------------------------------------------------------
# Artist slug
# ---------------------------------------------------------------------------


def artist_slug(artist: str) -> str:
    """Normalise an artist name to a directory slug."""
    slug = artist.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _extract_from_midi(midi_path: Path) -> Optional[dict]:
    """Extract raw statistical features from a single MIDI file.

    Returns a dict of raw measurements, or None if the file cannot be parsed.
    """
    try:
        mid = mido.MidiFile(str(midi_path))
    except Exception as exc:
        log.warning("Cannot parse MIDI %s: %s", midi_path.name, exc)
        return None

    tpb = mid.ticks_per_beat or 480

    # Merge all tracks into a single absolute-tick event stream
    events: list[dict] = []
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type in ("note_on", "note_off"):
                events.append(
                    {
                        "tick": tick,
                        "type": msg.type,
                        "note": msg.note,
                        "velocity": msg.velocity,
                    }
                )
    events.sort(key=lambda e: e["tick"])

    if not events:
        return None

    # Open/close notes
    open_notes: dict[int, tuple[int, int]] = {}  # note → (start_tick, velocity)
    notes: list[dict] = []
    for ev in events:
        if ev["type"] == "note_on" and ev["velocity"] > 0:
            open_notes[ev["note"]] = (ev["tick"], ev["velocity"])
        elif ev["type"] == "note_off" or (
            ev["type"] == "note_on" and ev["velocity"] == 0
        ):
            if ev["note"] in open_notes:
                start, vel = open_notes.pop(ev["note"])
                dur = ev["tick"] - start
                if dur > 0:
                    notes.append(
                        {
                            "note": ev["note"],
                            "start": start,
                            "dur": dur,
                            "velocity": vel,
                        }
                    )

    if not notes:
        return None

    total_ticks = max(n["start"] + n["dur"] for n in notes)
    bars = max(1, total_ticks / (tpb * 4))

    # Note density
    note_density = len(notes) / bars

    # Duration stats (in beats)
    durations = [n["dur"] / tpb for n in notes]
    mean_dur = sum(durations) / len(durations)
    dur_var = sum((d - mean_dur) ** 2 for d in durations) / len(durations)

    # Velocity stats
    velocities = [n["velocity"] for n in notes]
    vel_mean = sum(velocities) / len(velocities)
    vel_var = sum((v - vel_mean) ** 2 for v in velocities) / len(velocities)

    # Interval histogram (between consecutive notes by start time)
    sorted_notes = sorted(notes, key=lambda n: n["start"])
    intervals: list[int] = []
    for i in range(1, len(sorted_notes)):
        iv = sorted_notes[i]["note"] - sorted_notes[i - 1]["note"]
        intervals.append(max(-12, min(12, iv)))  # clamp to ±1 octave

    interval_counts: dict[int, int] = defaultdict(int)
    for iv in intervals:
        interval_counts[iv] += 1
    total_ivs = max(1, len(intervals))
    interval_histogram = {k: v / total_ivs for k, v in interval_counts.items()}

    # Rest ratio: fraction of bars with >50% silence
    bar_ticks = tpb * 4
    bar_count = max(1, int(total_ticks / bar_ticks) + 1)
    silent_bars = 0
    for b in range(bar_count):
        bar_start = b * bar_ticks
        bar_end = bar_start + bar_ticks
        occupied = sum(
            min(n["start"] + n["dur"], bar_end) - max(n["start"], bar_start)
            for n in notes
            if n["start"] < bar_end and n["start"] + n["dur"] > bar_start
        )
        if occupied < bar_ticks * 0.5:
            silent_bars += 1
    rest_ratio = silent_bars / bar_count

    # Harmonic rhythm: distinct pitch classes per bar
    bar_pitches: dict[int, set] = defaultdict(set)
    for n in notes:
        b = int(n["start"] / bar_ticks)
        bar_pitches[b].add(n["note"] % 12)
    harmonic_rhythm = (
        sum(len(v) for v in bar_pitches.values()) / len(bar_pitches)
        if bar_pitches
        else 0.0
    )

    # Phrase length: mean notes per silence-delimited phrase
    phrase_threshold = tpb * 2  # 2 beats silence = phrase boundary
    phrases: list[int] = []
    current_phrase = 1
    for i in range(1, len(sorted_notes)):
        gap = sorted_notes[i]["start"] - (
            sorted_notes[i - 1]["start"] + sorted_notes[i - 1]["dur"]
        )
        if gap > phrase_threshold:
            phrases.append(current_phrase)
            current_phrase = 1
        else:
            current_phrase += 1
    phrases.append(current_phrase)
    phrase_length_mean = sum(phrases) / len(phrases)

    return {
        "note_density": note_density,
        "note_density_variance": 0.0,  # per-file variance computed later
        "mean_duration_beats": mean_dur,
        "duration_variance": dur_var,
        "velocity_mean": vel_mean,
        "velocity_variance": vel_var,
        "interval_histogram": interval_histogram,
        "harmonic_rhythm": harmonic_rhythm,
        "rest_ratio": rest_ratio,
        "phrase_length_mean": phrase_length_mean,
    }


def extract_style_profile(
    midi_files: list[Path], artist: str = ""
) -> Optional[StyleProfile]:
    """Extract and average style features from a list of MIDI files.

    Returns None if no files could be parsed.
    """
    raw_profiles = [_extract_from_midi(f) for f in midi_files]
    valid = [p for p in raw_profiles if p is not None]
    if not valid:
        return None
    return _average_raw_profiles(valid, artist)


def _average_raw_profiles(profiles: list[dict], artist: str = "") -> StyleProfile:
    """Average a list of raw feature dicts into a single StyleProfile."""
    n = len(profiles)
    avg: dict = {}
    for key in (
        "note_density",
        "mean_duration_beats",
        "duration_variance",
        "velocity_mean",
        "velocity_variance",
        "harmonic_rhythm",
        "rest_ratio",
        "phrase_length_mean",
    ):
        avg[key] = sum(p[key] for p in profiles) / n

    # Variance of note_density across files
    nd = avg["note_density"]
    avg["note_density_variance"] = (
        sum((p["note_density"] - nd) ** 2 for p in profiles) / n
    )

    # Merge interval histograms
    merged: dict[int, float] = defaultdict(float)
    for p in profiles:
        for k, v in p.get("interval_histogram", {}).items():
            merged[k] += v
    total = sum(merged.values()) or 1.0
    avg["interval_histogram"] = {k: v / total for k, v in merged.items()}

    return StyleProfile(artist=artist, **avg)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

PROFILE_FILENAME = "profile.yml"


def _profile_is_stale(profile_path: Path, midi_files: list[Path]) -> bool:
    """Return True if the cached profile is older than any source MIDI file."""
    if not profile_path.exists():
        return True
    profile_mtime = profile_path.stat().st_mtime
    return any(f.stat().st_mtime > profile_mtime for f in midi_files)


def load_or_extract_profile(
    artist: str,
    style_refs_dir: Path | None = None,
) -> Optional[StyleProfile]:
    """Load a cached StyleProfile or extract it fresh from MIDI files.

    Returns None if no MIDI files are found for the artist.
    """
    refs_dir = Path(style_refs_dir) if style_refs_dir else DEFAULT_STYLE_REFS_DIR
    slug = artist_slug(artist)
    artist_dir = refs_dir / slug

    if not artist_dir.is_dir():
        log.debug("No style_references directory for %r (%s)", artist, artist_dir)
        return None

    midi_files = list(artist_dir.glob("*.mid")) + list(artist_dir.glob("*.midi"))
    if not midi_files:
        log.warning("No MIDI files found for %r in %s", artist, artist_dir)
        return None

    profile_path = artist_dir / PROFILE_FILENAME

    if not _profile_is_stale(profile_path, midi_files):
        try:
            data = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
            return StyleProfile(**data)
        except Exception as exc:
            log.warning("Failed to load cached profile for %r: %s", artist, exc)

    # Extract fresh
    profile = extract_style_profile(midi_files, artist=artist)
    if profile is None:
        log.warning("No valid MIDI data extracted for %r", artist)
        return None

    # Cache
    profile_path.write_text(
        yaml.dump(
            profile.model_dump(),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    log.info("Profile extracted and cached: %s", profile_path)
    return profile


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def aggregate_profiles(profiles: list[StyleProfile]) -> Optional[StyleProfile]:
    """Average a list of StyleProfiles into a single aggregate profile."""
    if not profiles:
        return None
    raw = [p.model_dump() for p in profiles]
    # Remove non-numeric fields before averaging
    numeric_keys = [
        "note_density",
        "note_density_variance",
        "mean_duration_beats",
        "duration_variance",
        "velocity_mean",
        "velocity_variance",
        "harmonic_rhythm",
        "rest_ratio",
        "phrase_length_mean",
    ]
    n = len(raw)
    avg: dict = {}
    for k in numeric_keys:
        avg[k] = sum(p[k] for p in raw) / n

    # Merge interval histograms
    merged: dict[int, float] = defaultdict(float)
    for p in raw:
        for k, v in (p.get("interval_histogram") or {}).items():
            merged[k] += v
    total = sum(merged.values()) or 1.0
    avg["interval_histogram"] = {k: v / total for k, v in merged.items()}
    avg["style_weight"] = sum(p.get("style_weight", 0.4) for p in raw) / n
    avg["artist"] = ", ".join(p["artist"] for p in raw if p.get("artist"))

    return StyleProfile(**avg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="MIDI Style Reference utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pop = sub.add_parser(
        "populate", help="Copy MIDI files and extract profile for an artist"
    )
    pop.add_argument("--artist", required=True)
    pop.add_argument("--files", required=True, nargs="+", help="MIDI files to copy")
    pop.add_argument(
        "--style-refs-dir",
        default=None,
        help="Path to style_references/ directory (default: project root)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.cmd == "populate":
        refs_dir = (
            Path(args.style_refs_dir) if args.style_refs_dir else DEFAULT_STYLE_REFS_DIR
        )
        slug = artist_slug(args.artist)
        dest = refs_dir / slug
        dest.mkdir(parents=True, exist_ok=True)

        copied = []
        for src in args.files:
            src_path = Path(src)
            if not src_path.exists():
                print(f"  SKIP: {src} (not found)")
                continue
            out = dest / src_path.name
            shutil.copy2(src_path, out)
            copied.append(out)
            print(f"  Copied: {src_path.name}")

        if not copied:
            print("No files copied — nothing to extract.")
            sys.exit(1)

        profile = extract_style_profile(copied, artist=args.artist)
        if profile is None:
            print("Profile extraction failed.")
            sys.exit(1)

        profile_path = dest / PROFILE_FILENAME
        profile_path.write_text(
            yaml.dump(
                profile.model_dump(),
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        print(f"\nProfile written: {profile_path}")
        print(f"  note_density     : {profile.note_density:.2f} notes/bar")
        print(f"  mean_duration    : {profile.mean_duration_beats:.2f} beats")
        print(f"  velocity_mean    : {profile.velocity_mean:.1f}")
        print(f"  rest_ratio       : {profile.rest_ratio:.2f}")
        print(f"  harmonic_rhythm  : {profile.harmonic_rhythm:.2f} pitch classes/bar")


if __name__ == "__main__":
    main()
