#!/usr/bin/env python3
"""
Assembly manifest importer for the Music Production Pipeline.

Parses a Logic Pro arrangement export (timecode format) and updates
production_plan.yml + manifest_bootstrap.yml with actual section boundaries,
loop assignments, and vocals flags. Emits drift_report.yml showing deviation
from the computed plan.

Usage:
    python -m app.generators.midi.assembly_manifest \
        --production-dir shrinkwrapped/.../production/black__sequential_dissolution_v2 \
        --arrangement archivist_arrangement.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from app.generators.midi.production_plan import (
    MANIFEST_BOOTSTRAP_FILENAME,
    PLAN_FILENAME,
    load_plan,
    save_plan,
)

DRIFT_REPORT_FILENAME = "drift_report.yml"

SECTION_PREFIXES = {
    "intro": "Intro",
    "verse": "Verse",
    "bridge": "Bridge",
    "outro": "Outro",
    "chorus": "Chorus",
}

# Instrument prefixes stripped when matching loop names against known section names
_INSTRUMENT_PREFIXES = ("melody_", "bass_", "drums_", "chords_")
_TRAILING_SUFFIX_RE = re.compile(r"(_alt|_\d+)+$")

DEFAULT_TRACK_MAP: dict[int, str] = {
    1: "chords",
    2: "drums",
    3: "bass",
    4: "melody",
}

# Approved subfolders that correspond to instrument families
INSTRUMENT_FOLDERS = ("chords", "drums", "bass", "melody", "strums", "harmonic_rhythm")


def build_folder_lookup(production_dir: Path) -> dict[str, str]:
    """Scan approved subfolders and return {loop_stem: instrument} lookup.

    When a loop name appears in multiple approved folders the first match
    in INSTRUMENT_FOLDERS order wins (chords > drums > bass > melody).
    """
    lookup: dict[str, str] = {}
    for instrument in INSTRUMENT_FOLDERS:
        approved = production_dir / instrument / "approved"
        if not approved.is_dir():
            continue
        for mid in approved.glob("*.mid"):
            stem = mid.stem
            if stem not in lookup:
                lookup[stem] = instrument
    return lookup


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Clip:
    start: float  # seconds from arrangement start
    name: str
    track: int
    length: float  # seconds


@dataclass
class ArrangementSection:
    name: str  # derived from loop prefix ("Intro", "Verse", …)
    start: float  # seconds
    end: float  # seconds
    vocals: bool = False
    loops: dict = field(default_factory=dict)  # {instrument: loop_name}


@dataclass
class DriftEntry:
    section_index: int
    plan_name: str
    arrangement_name: str
    computed_start: float
    actual_start: float
    drift_seconds: float
    vocals_flag_changed: bool


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _tc_to_seconds(tc: str) -> float:
    """Parse Logic Pro timecode HH:MM:SS:FF.sub → total seconds.

    Ignores frames and sub-frames.  Handles 4-part (HH:MM:SS:FF),
    3-part (HH:MM:SS), and 2-part (MM:SS) variants.
    """
    parts = tc.strip().split(":")
    try:
        if len(parts) >= 4:
            h = int(parts[0])
            m = int(parts[1])
            s = int(parts[2].split(".")[0])
        elif len(parts) == 3:
            h = int(parts[0])
            m = int(parts[1])
            s = int(parts[2].split(".")[0])
        elif len(parts) == 2:
            h = 0
            m = int(parts[0])
            s = int(parts[1].split(".")[0])
        else:
            return 0.0
    except ValueError:
        return 0.0
    return h * 3600.0 + m * 60.0 + s


def parse_arrangement(text: str) -> list[Clip]:
    """Parse Logic Pro arrangement export text → list of Clips.

    Expected format per non-empty line:
        HH:MM:SS:FF.sub   loop_name   track_number   HH:MM:SS:FF.sub

    The first position encountered is used as the base offset so that
    clip starts are relative to the song start (handles Logic's 01:00:00:00
    hour offset automatically).
    """
    clips: list[Clip] = []
    base_offset: Optional[float] = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            # Field order: start_tc  name  track  length_tc
            # Name may contain spaces — parse from both ends.
            start_secs = _tc_to_seconds(parts[0])
            length_secs = _tc_to_seconds(parts[-1])
            track = int(parts[-2])
            name = " ".join(parts[1:-2])
        except (ValueError, IndexError):
            continue

        if base_offset is None:
            base_offset = start_secs

        # Detect format: Logic exports two variants —
        #   duration format: "00:00:10:00.00" → length_secs = 10.0 (< start_secs)
        #   end-position format: "01:00:12:00.00" → length_secs >= start_secs, subtract to get duration
        if length_secs >= start_secs:
            clip_length = round(length_secs - start_secs, 3)
        else:
            clip_length = round(length_secs, 3)

        clips.append(
            Clip(
                start=round(start_secs - base_offset, 3),
                name=name,
                track=track,
                length=clip_length,
            )
        )

    return clips


# ---------------------------------------------------------------------------
# Section derivation
# ---------------------------------------------------------------------------


def _section_name_from_loop(
    loop_name: str,
    known_sections: Optional[frozenset] = None,
) -> str:
    """Derive canonical section name from a loop name.

    When *known_sections* is provided (a frozenset of lowercase plan section
    names) the function first tries a direct or prefix-stripped match against
    those names before falling back to SECTION_PREFIXES.
    """
    lower = loop_name.strip().lower()

    if known_sections:
        # Direct match
        if lower in known_sections:
            return lower
        # Strip one instrument prefix, then try direct + suffix-stripped match
        for prefix in _INSTRUMENT_PREFIXES:
            if lower.startswith(prefix):
                stripped = lower[len(prefix) :]
                if stripped in known_sections:
                    return stripped
                base = _TRAILING_SUFFIX_RE.sub("", stripped)
                if base in known_sections:
                    return base
                break  # only strip the first matching instrument prefix

    # Fall back to SECTION_PREFIXES (Intro/Verse/Bridge/etc.)
    for prefix, canonical in SECTION_PREFIXES.items():
        if (
            lower.startswith(prefix + "_")
            or lower.startswith(prefix + " ")
            or lower == prefix
        ):
            return canonical
    return "unknown"


def derive_sections(
    clips: list[Clip],
    track_map: Optional[dict[int, str]] = None,
    vocalist_suffix: str = "_gw",
    folder_lookup: Optional[dict[str, str]] = None,
    known_sections: Optional[frozenset] = None,
) -> list[ArrangementSection]:
    """Group clips into named sections based on loop name changes.

    A new section begins whenever the dominant loop name changes between
    consecutive time slots (all clips sharing the same start time).

    When *known_sections* is provided (a frozenset of lowercase plan section
    names), section changes are only triggered by clips on designated chords
    tracks.  This avoids spurious section boundaries caused by bass or drum
    loops that overlap neighbouring sections.

    Args:
        clips: Output of parse_arrangement().
        track_map: Maps track number → instrument name.
        vocalist_suffix: Loop name suffix that marks a sung melody track.
        folder_lookup: {loop_stem: instrument} from approved sub-folders.
        known_sections: Frozenset of lowercase plan section names.  When
            supplied, loop names are matched against these before falling back
            to the built-in SECTION_PREFIXES.

    Returns:
        Ordered list of ArrangementSection objects.
    """
    if not clips:
        return []

    if track_map is None:
        track_map = DEFAULT_TRACK_MAP

    # Identify which tracks carry chords (primary section-change triggers)
    chords_track_nums: frozenset[int] = frozenset(
        t for t, instr in track_map.items() if instr == "chords"
    )

    # Group clips by start time → time slots
    slots: dict[float, list[Clip]] = {}
    for clip in clips:
        slots.setdefault(clip.start, []).append(clip)
    sorted_starts = sorted(slots.keys())

    def _slot_name(slot_clips: list[Clip]) -> Optional[str]:
        """Return the section name for this slot, or None to inherit.

        When *known_sections* is set and chords tracks are defined in
        track_map, only a clip on a chords track can trigger a new section.
        If no qualifying chords clip is present, returns None so the caller
        inherits the current section (avoiding false boundaries from
        secondary-instrument loops that extend across section edges).
        """
        if known_sections and chords_track_nums:
            for track_num in sorted(chords_track_nums):
                for c in slot_clips:
                    if c.track == track_num:
                        n = _section_name_from_loop(c.name, known_sections)
                        if n != "unknown":
                            return n
            # No chords clip with a recognisable name → inherit
            return None

        # Original fallback: try tracks in ascending order, then any track
        for track_num in sorted(track_map.keys()):
            for c in slot_clips:
                if c.track == track_num:
                    n = _section_name_from_loop(c.name, known_sections)
                    if n != "unknown":
                        return n
        for c in sorted(slot_clips, key=lambda x: x.track):
            n = _section_name_from_loop(c.name, known_sections)
            if n != "unknown":
                return n
        return "unknown"

    def _accumulate(slot_clips: list[Clip]) -> None:
        nonlocal current_vocals
        for clip in slot_clips:
            if folder_lookup and clip.name in folder_lookup:
                instrument = folder_lookup[clip.name]
            else:
                instrument = track_map.get(clip.track, f"track_{clip.track}")
            # First occurrence wins: the opening clip for each instrument in a
            # section defines the canonical loop.  This prevents overlapping
            # secondary tracks (e.g. a honey bass loop extending into toppling)
            # from overwriting the section's true instrument assignment.
            if instrument not in current_loops:
                current_loops[instrument] = clip.name
            if instrument == "melody":
                if (vocalist_suffix and clip.name.endswith(vocalist_suffix)) or (
                    "vocal" in clip.name.lower()
                ):
                    current_vocals = True

    sections: list[ArrangementSection] = []
    current_name: Optional[str] = None
    current_start: Optional[float] = None
    current_loops: dict[str, str] = {}
    current_vocals: bool = False
    warned_unknown: set[str] = set()

    for slot_start in sorted_starts:
        slot_clips = slots[slot_start]
        slot_name = _slot_name(slot_clips)

        # None → inherit current section; just accumulate loops
        if slot_name is None:
            _accumulate(slot_clips)
            continue

        if slot_name == "unknown":
            for c in slot_clips:
                if c.name not in warned_unknown:
                    print(
                        f"  Warning: unrecognised loop prefix '{c.name}'",
                        file=sys.stderr,
                    )
                    warned_unknown.add(c.name)

        if slot_name != current_name:
            # Flush accumulated section
            if current_name is not None:
                sections.append(
                    ArrangementSection(
                        name=current_name,
                        start=current_start,
                        end=slot_start,
                        vocals=current_vocals,
                        loops=current_loops,
                    )
                )
            current_name = slot_name
            current_start = slot_start
            current_loops = {}
            current_vocals = False

        _accumulate(slot_clips)

    # Flush final section
    if current_name is not None:
        last_clips = slots[sorted_starts[-1]]
        # clip.length is always a duration; add to the last slot's start
        end_time = sorted_starts[-1] + max(c.length for c in last_clips)
        sections.append(
            ArrangementSection(
                name=current_name,
                start=current_start,
                end=round(end_time, 3),
                vocals=current_vocals,
                loops=current_loops,
            )
        )

    return sections


# ---------------------------------------------------------------------------
# Drift computation
# ---------------------------------------------------------------------------


def _fmt_time(t: float) -> str:
    m = int(t // 60)
    s = t % 60
    return f"[{m:02d}:{s:06.3f}]"


def _computed_section_start(plan, section_index: int) -> float:
    """Return the computed start time (seconds) for a section by index."""
    parts = plan.time_sig.split("/")
    beats_per_bar = int(parts[0]) * (4.0 / int(parts[1]))
    spb = beats_per_bar * (60.0 / plan.bpm)
    cursor = 0.0
    for i, sec in enumerate(plan.sections):
        if i == section_index:
            return cursor
        cursor += sec.bars * sec.repeat * spb
    return cursor


def compute_drift(plan, actual_sections: list[ArrangementSection]) -> list[DriftEntry]:
    """Compare plan-computed section starts against actual arrangement times.

    Matches each actual section to the first unmatched plan section with the
    same name, then reports timing delta against that section's computed start.
    Positional matching was incorrect when the arrangement has fewer entries
    than the plan (e.g. user arranged fewer repetitions than planned).
    """
    entries: list[DriftEntry] = []
    plan_matched = [False] * len(plan.sections)

    for actual in actual_sections:
        # Find first unmatched plan section with matching name
        matched_idx: Optional[int] = None
        for i, sec in enumerate(plan.sections):
            if not plan_matched[i] and sec.name == actual.name:
                matched_idx = i
                break

        if matched_idx is not None:
            plan_matched[matched_idx] = True
            computed = _computed_section_start(plan, matched_idx)
            plan_sec = plan.sections[matched_idx]
        else:
            # Section in arrangement has no matching plan entry — no drift to report
            computed = actual.start
            plan_sec = None

        plan_name = plan_sec.name if plan_sec else actual.name
        vocals_changed = plan_sec is not None and plan_sec.vocals != actual.vocals
        entries.append(
            DriftEntry(
                section_index=len(entries),
                plan_name=plan_name,
                arrangement_name=actual.name,
                computed_start=round(computed, 3),
                actual_start=round(actual.start, 3),
                drift_seconds=round(actual.start - computed, 3),
                vocals_flag_changed=vocals_changed,
            )
        )
    return entries


# ---------------------------------------------------------------------------
# Plan + manifest update
# ---------------------------------------------------------------------------


def _seconds_per_bar(plan) -> float:
    parts = plan.time_sig.split("/")
    beats_per_bar = int(parts[0]) * (4.0 / int(parts[1]))
    return beats_per_bar * (60.0 / plan.bpm)


def _update_plan(plan, actual_sections: list[ArrangementSection]) -> None:
    """Mutate plan sections with actual bar counts, loops, and vocals.

    Matches detected arrangement sections to plan sections by name (first
    unmatched wins).  Loop assignments are then propagated to any remaining
    plan sections that share the same name but had no direct arrangement match
    (e.g. repeated sections that the user arranged only once in the loop grid).
    """
    spb = _seconds_per_bar(plan)
    plan_matched = [False] * len(plan.sections)
    # section_name → loops dict from the first matched arrangement section
    name_to_loops: dict[str, dict] = {}

    for actual in actual_sections:
        for i, sec in enumerate(plan.sections):
            if plan_matched[i]:
                continue
            if sec.name != actual.name:
                continue
            plan_matched[i] = True
            duration = actual.end - actual.start
            total_bars = max(1, round(duration / spb))
            sec.bars = max(1, round(total_bars / max(sec.repeat, 1)))
            sec.loops = dict(actual.loops)
            if not sec.vocals:
                sec.vocals = actual.vocals
            # Record loops for propagation to other sections with same name
            if actual.name not in name_to_loops:
                name_to_loops[actual.name] = dict(actual.loops)
            break

    # Propagate loop assignments to unmatched plan sections with the same name
    for i, sec in enumerate(plan.sections):
        if not plan_matched[i] and sec.name in name_to_loops:
            sec.loops = dict(name_to_loops[sec.name])

    plan.vocals_planned = any(s.vocals for s in plan.sections)


def _update_manifest(production_dir: Path, plan) -> None:
    """Rewrite manifest_bootstrap.yml structure timestamps from updated plan."""
    manifest_path = production_dir / MANIFEST_BOOTSTRAP_FILENAME
    if not manifest_path.exists():
        return
    with open(manifest_path) as f:
        data = yaml.safe_load(f)

    spb = _seconds_per_bar(plan)
    cursor = 0.0
    structure = []
    for sec in plan.sections:
        duration = sec.bars * sec.repeat * spb
        end = cursor + duration
        structure.append(
            {
                "section_name": sec.name,
                "start_time": _fmt_time(cursor),
                "end_time": _fmt_time(end),
                "description": sec.notes or None,
            }
        )
        cursor = end

    data["structure"] = structure
    data["vocals"] = any(s.vocals for s in plan.sections)

    with open(manifest_path, "w") as f:
        yaml.dump(
            data, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )


def _write_drift_report(
    production_dir: Path,
    source_arrangement: str,
    drift_entries: list[DriftEntry],
) -> Path:
    entries = []
    for e in drift_entries:
        entry: dict = {
            "section_index": e.section_index,
            "plan_name": e.plan_name,
            "arrangement_name": e.arrangement_name,
            "computed_start": _fmt_time(e.computed_start),
            "actual_start": _fmt_time(e.actual_start),
            "drift_seconds": e.drift_seconds,
        }
        if e.vocals_flag_changed:
            entry["vocals_flag_changed"] = True
        if e.plan_name != e.arrangement_name:
            entry["name_mismatch"] = True
        entries.append(entry)

    data = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source_arrangement": source_arrangement,
        "sections": entries,
    }
    out_path = production_dir / DRIFT_REPORT_FILENAME
    with open(out_path, "w") as f:
        yaml.dump(
            data, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )
    return out_path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def import_arrangement(
    production_dir: Path,
    arrangement_path: Path,
    track_map: Optional[dict[int, str]] = None,
    vocalist_suffix: str = "_gw",
) -> list[DriftEntry]:
    """Parse Logic arrangement, update plan + manifest, return drift.

    Raises FileNotFoundError if arrangement_path or production_plan.yml absent.
    """
    if not arrangement_path.exists():
        raise FileNotFoundError(f"Arrangement file not found: {arrangement_path}")

    plan = load_plan(production_dir)
    if plan is None:
        raise FileNotFoundError(
            f"No {PLAN_FILENAME} found in {production_dir}. Generate a plan first."
        )

    text = arrangement_path.read_text()
    clips = parse_arrangement(text)
    folder_lookup = build_folder_lookup(production_dir)
    known_sections: Optional[frozenset] = frozenset(
        s.name.lower() for s in plan.sections
    )
    actual_sections = derive_sections(
        clips,
        track_map or DEFAULT_TRACK_MAP,
        vocalist_suffix,
        folder_lookup,
        known_sections,
    )

    drift = compute_drift(plan, actual_sections)
    _update_plan(plan, actual_sections)
    save_plan(plan, production_dir)
    _update_manifest(production_dir, plan)
    _write_drift_report(production_dir, str(arrangement_path), drift)

    return drift


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_track_map(s: str) -> dict[int, str]:
    result = {}
    for part in s.split(","):
        k, v = part.strip().split("=")
        result[int(k.strip())] = v.strip()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import Logic Pro arrangement into production plan and manifest"
    )
    parser.add_argument("--production-dir", required=True)
    parser.add_argument(
        "--arrangement", required=True, help="Path to Logic arrangement export file"
    )
    parser.add_argument(
        "--track-map",
        default="1=chords,2=drums,3=bass,4=melody",
        help="Track number → instrument mapping (default: 1=chords,2=drums,3=bass,4=melody)",
    )
    parser.add_argument(
        "--vocalist-suffix",
        default="_gw",
        help="Loop name suffix indicating a sung melody (default: _gw)",
    )
    args = parser.parse_args()

    prod_path = Path(args.production_dir)
    arr_path = Path(args.arrangement)

    if not prod_path.exists():
        print(f"ERROR: Production directory not found: {prod_path}")
        sys.exit(1)

    try:
        track_map = _parse_track_map(args.track_map)
    except ValueError:
        print(f"ERROR: Invalid --track-map format: {args.track_map}")
        sys.exit(1)

    print("=" * 60)
    print("ASSEMBLY MANIFEST IMPORT")
    print("=" * 60)

    try:
        drift = import_arrangement(prod_path, arr_path, track_map, args.vocalist_suffix)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    any_drift = False
    print(f"\nSections found: {len(drift)}")
    for e in drift:
        drift_str = f"{e.drift_seconds:+.1f}s" if e.drift_seconds != 0 else "no drift"
        name_note = (
            f" [was '{e.plan_name}']" if e.plan_name != e.arrangement_name else ""
        )
        vocals_note = " [vocals flag updated]" if e.vocals_flag_changed else ""
        print(
            f"  {e.section_index + 1}. {e.arrangement_name:<12}"
            f"  actual {_fmt_time(e.actual_start)}  {drift_str}{name_note}{vocals_note}"
        )
        if e.drift_seconds != 0:
            any_drift = True

    print()
    print(f"production_plan.yml updated: {prod_path / PLAN_FILENAME}")
    print(f"manifest_bootstrap.yml updated: {prod_path / MANIFEST_BOOTSTRAP_FILENAME}")
    if any_drift:
        print(f"drift_report.yml written: {prod_path / DRIFT_REPORT_FILENAME}")


if __name__ == "__main__":
    main()
