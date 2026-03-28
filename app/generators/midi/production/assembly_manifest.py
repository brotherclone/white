#!/usr/bin/env python3
"""
Assembly manifest importer for the Music Production Pipeline.

Parses a Logic Pro arrangement export (timecode format) and updates
production_plan.yml + manifest_bootstrap.yml with actual section boundaries,
loop assignments, and vocals flags. Emits drift_report.yml showing deviation
from the computed plan.

Usage:
    python -m app.generators.midi.production.assembly_manifest \
        --production-dir shrink_wrapped/.../production/black__sequential_dissolution_v2 \
        --arrangement archivist_arrangement.txt
"""

from __future__ import annotations

import argparse
import re
import sys
import mido
import yaml

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.generators.midi.production.production_plan import (
    MANIFEST_BOOTSTRAP_FILENAME,
    PLAN_FILENAME,
    load_plan,
    load_song_proposal,
    save_plan,
)

DRIFT_REPORT_FILENAME = "drift_report.yml"
TRACK_MANIFEST_FILENAME = "track_manifest.yml"

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


def _bars_beats_to_seconds(
    bar: int,
    beat: int,
    subdiv: int,
    tick: int,
    bpm: float,
    beats_per_bar: int,
) -> float:
    """Convert Logic Pro bar/beat/subdivision/tick position to seconds.

    Logic displays positions as "Bar Beat Subdiv Tick" where:
    - Bar and Beat are 1-indexed
    - Subdiv is the 1/16-note position within the beat (1–4 in 4/4)
    - Tick is the fine-grained position within the subdivision (1–240 typical)
    """
    total_beats = (bar - 1) * beats_per_bar + (beat - 1)
    frac_beats = (subdiv - 1) / 4.0 + (tick - 1) / (4.0 * 240.0)
    return (total_beats + frac_beats) * (60.0 / bpm)


def _is_bar_beat_format(text: str) -> bool:
    """Return True if text looks like Logic's bar/beat position format.

    Timecode lines start with a token containing colons (``01:00:00:00.00``).
    Bar/beat lines start with a bare integer (``1``, ``13``, …).
    Returns True only when the first recognisable line begins with a pure integer.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        first_token = stripped.split()[0] if stripped.split() else ""
        if not first_token:
            continue
        # Bar/beat: pure integer (e.g. "1", "13")
        # Timecode: contains colons (e.g. "01:00:00:00.00")
        return first_token.isdigit()
    return False


def _parse_bar_position(pos_str: str) -> tuple[int, int, int, int]:
    """Parse a Logic bar/beat position string '1 1 1 1' → (bar, beat, subdiv, tick)."""
    parts = pos_str.strip().split()
    bar = int(parts[0]) if len(parts) > 0 else 1
    beat = int(parts[1]) if len(parts) > 1 else 1
    subdiv = int(parts[2]) if len(parts) > 2 else 1
    tick = int(parts[3]) if len(parts) > 3 else 1
    return bar, beat, subdiv, tick


def parse_arrangement_bars_beats(
    text: str,
    bpm: float = 120.0,
    beats_per_bar: int = 4,
) -> list[Clip]:
    """Parse Logic Pro bar/beat format arrangement export → list of Clips.

    Expected format per non-empty line (tab-delimited):
        BAR BEAT SUBDIV TICK \\t loop_name \\t track_number \\t BAR BEAT SUBDIV TICK

    The first position encountered is used as the base offset.
    """
    clips: list[Clip] = []
    base_offset: Optional[float] = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        fields = [f.strip() for f in stripped.split("\t")]
        if len(fields) < 4:
            continue
        try:
            s_bar, s_beat, s_sub, s_tick = _parse_bar_position(fields[0])
            e_bar, e_beat, e_sub, e_tick = _parse_bar_position(fields[3])
            name = fields[1]
            track = int(fields[2])
        except (ValueError, IndexError):
            continue

        start_secs = _bars_beats_to_seconds(
            s_bar, s_beat, s_sub, s_tick, bpm, beats_per_bar
        )
        end_secs = _bars_beats_to_seconds(
            e_bar, e_beat, e_sub, e_tick, bpm, beats_per_bar
        )

        if base_offset is None:
            base_offset = start_secs

        clip_length = round(end_secs - start_secs, 3)
        clips.append(
            Clip(
                start=round(start_secs - base_offset, 3),
                name=name,
                track=track,
                length=clip_length,
            )
        )

    return clips


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


def parse_arrangement(
    text: str,
    bpm: float = 120.0,
    beats_per_bar: int = 4,
) -> list[Clip]:
    """Parse Logic Pro arrangement export text → list of Clips.

    Supports two Logic export formats automatically:
    - **Timecode** (``01:00:00:00.00  loop_name  track  00:00:10:00.00``):
      the classic HH:MM:SS:FF format exported from Logic's timecode display.
    - **Bar/beat** (tab-delimited ``1 1 1 1 \\t loop_name \\t track \\t 7 1 1 1``):
      Logic's bar-and-beats position display.  Requires *bpm* and *beats_per_bar*
      to convert positions to seconds.

    The first position encountered is used as the base offset so that clip
    starts are relative to the song start.
    """
    if _is_bar_beat_format(text):
        return parse_arrangement_bars_beats(text, bpm=bpm, beats_per_bar=beats_per_bar)
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

        # None → inherit current section; just accumulate loops.
        # Exception: if no section has started yet (current_name is None), fall back
        # to any recognisable non-chords track so that sections that have no chords
        # track at all (e.g. a drums-only intro) are still detected.
        if slot_name is None:
            if current_name is None:
                for c in sorted(slot_clips, key=lambda x: x.track):
                    n = _section_name_from_loop(c.name, known_sections)
                    if n != "unknown":
                        slot_name = n
                        break
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
        cursor += sec.bars * sec.play_count * spb
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
            sec.bars = max(1, round(total_bars / max(sec.play_count, 1)))
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
        duration = sec.bars * sec.play_count * spb
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


def compute_proposal_drift(
    proposal_path: Path,
    actual_sections: list[ArrangementSection],
) -> Optional[dict]:
    """Compare a composition_proposal.yml against actual arrangement sections.

    Returns a dict with:
      sections_added   — names in arrangement but absent from proposal
      sections_removed — names in proposal but absent from arrangement
      play_count_deltas    — [{name, proposed, actual}] where repeat count changed
      order_changed    — True if shared section ordering differs

    Returns None if the proposal file cannot be loaded or has no proposed_sections.
    """
    if not proposal_path.exists():
        return None
    try:
        with open(proposal_path) as f:
            proposal = yaml.safe_load(f) or {}
    except Exception:
        return None

    proposed_sections = proposal.get("proposed_sections") or []
    if not proposed_sections:
        return None

    proposed_names = [s.get("name", "") for s in proposed_sections]
    proposed_repeats = {
        s.get("name", ""): int(s.get("play_count", s.get("repeat", 1)))
        for s in proposed_sections
    }
    actual_names = [s.name for s in actual_sections]
    actual_repeats: dict[str, int] = {}
    for s in actual_sections:
        actual_repeats[s.name] = actual_repeats.get(s.name, 0) + 1

    proposed_set = set(proposed_names)
    actual_set = set(actual_names)

    sections_added = sorted(actual_set - proposed_set)
    sections_removed = sorted(proposed_set - actual_set)

    play_count_deltas = []
    for name in proposed_set & actual_set:
        p_repeat = proposed_repeats.get(name, 1)
        a_repeat = actual_repeats.get(name, 1)
        if p_repeat != a_repeat:
            play_count_deltas.append(
                {"name": name, "proposed": p_repeat, "actual": a_repeat}
            )
    play_count_deltas.sort(key=lambda x: x["name"])

    # Order changed: compare relative order of shared sections
    shared = proposed_set & actual_set
    proposed_order = [n for n in proposed_names if n in shared]
    actual_order = [n for n in actual_names if n in shared]
    # Deduplicate while preserving order for comparison
    seen: set = set()
    proposed_dedup = []
    for n in proposed_order:
        if n not in seen:
            proposed_dedup.append(n)
            seen.add(n)
    seen = set()
    actual_dedup = []
    for n in actual_order:
        if n not in seen:
            actual_dedup.append(n)
            seen.add(n)
    order_changed = proposed_dedup != actual_dedup

    return {
        "sections_added": sections_added,
        "sections_removed": sections_removed,
        "play_count_deltas": play_count_deltas,
        "order_changed": order_changed,
    }


def _write_drift_report(
    production_dir: Path,
    source_arrangement: str,
    drift_entries: list[DriftEntry],
    proposal_drift: Optional[dict] = None,
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
    if proposal_drift is not None:
        data["proposal_drift"] = proposal_drift
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
    time_sig_parts = plan.time_sig.split("/")
    beats_per_bar = (
        int(time_sig_parts[0])
        * (4.0 / int(time_sig_parts[1]))
        * (4.0 / int(time_sig_parts[1]))
    )
    clips = parse_arrangement(text, bpm=float(plan.bpm), beats_per_bar=beats_per_bar)
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

    from app.generators.midi.production.composition_proposal import PROPOSAL_FILENAME

    proposal_drift = compute_proposal_drift(
        production_dir / PROPOSAL_FILENAME, actual_sections
    )
    _write_drift_report(production_dir, str(arrangement_path), drift, proposal_drift)

    return drift


# ---------------------------------------------------------------------------
# Track manifest generation
# ---------------------------------------------------------------------------


def _resolve_proposal_path(
    production_dir: Path, source_proposal: str
) -> Optional[Path]:
    """Try to resolve source_proposal relative to project root.

    Walks up from production_dir trying each ancestor as project root.
    Also tries inserting 'yml/' before the filename, since some songs store
    proposals under a yml/ subdirectory within the thread.
    """
    sp = Path(source_proposal)
    candidate = production_dir
    for _ in range(10):
        candidate = candidate.parent
        direct = candidate / sp
        if direct.exists():
            return direct
        # Try with 'yml/' inserted before the filename
        with_yml = candidate / sp.parent / "yml" / sp.name
        if with_yml.exists():
            return with_yml
    return None


def generate_track_manifest(
    production_dir: Path,
    arrangement_path: Path,
    song_proposal_path: Optional[Path] = None,
    track_map: Optional[dict[int, str]] = None,
    vocalist_suffix: str = "_gw",
) -> Path:
    """Generate track_manifest.yml from arrangement.txt + song proposal.

    Reads arrangement clips, derives sections, and combines with song proposal
    identity fields to produce a canonical track manifest. This replaces
    manifest_bootstrap.yml as the bridge between the generative pipeline and
    release. arrangement.txt is the authoritative source for structure; the
    song proposal YAML is the authoritative source for song identity.

    Args:
        production_dir: Song production directory (must contain production_plan.yml).
        arrangement_path: Path to Logic Pro arrangement export file.
        song_proposal_path: Path to song proposal YAML. Auto-detected from
            plan.source_proposal if not supplied.
        track_map: {track_number: instrument_folder}. Defaults to DEFAULT_TRACK_MAP.
        vocalist_suffix: Loop name suffix indicating a sung melody (default: _gw).

    Returns:
        Path to the written track_manifest.yml.
    """
    plan = load_plan(production_dir)
    if plan is None:
        raise FileNotFoundError(
            f"No {PLAN_FILENAME} found in {production_dir}. Generate a plan first."
        )

    if track_map is None:
        track_map = DEFAULT_TRACK_MAP

    # Parse arrangement → clips
    text = arrangement_path.read_text()
    time_sig_parts = plan.time_sig.split("/")
    beats_per_bar = int(time_sig_parts[0]) * (4.0 / int(time_sig_parts[1]))
    clips = parse_arrangement(text, bpm=float(plan.bpm), beats_per_bar=beats_per_bar)

    # Find song proposal (auto-detect from plan if not supplied)
    if song_proposal_path is None and plan.source_proposal:
        song_proposal_path = _resolve_proposal_path(
            production_dir, plan.source_proposal
        )

    # Load proposal data — raw YAML (for singer field) + normalised dict
    proposal_raw: dict = {}
    proposal_data: dict = {}
    if song_proposal_path is not None and song_proposal_path.exists():
        with open(song_proposal_path) as f:
            proposal_raw = yaml.safe_load(f)
        proposal_data = load_song_proposal(song_proposal_path)

    # Derive sections from clips
    folder_lookup = build_folder_lookup(production_dir)
    known_sections: Optional[frozenset] = frozenset(
        s.name.lower() for s in plan.sections
    )
    sections = derive_sections(
        clips, track_map, vocalist_suffix, folder_lookup, known_sections
    )

    # Render-time top-level flags
    vocals = any(s.vocals for s in sections)
    lyrics_path = production_dir / "melody" / "lyrics.txt"
    lyrics = lyrics_path.exists()

    # Singer lives in raw proposal but is not surfaced by load_song_proposal()
    singer = proposal_raw.get("singer") if proposal_raw else None

    data = {
        "manifest_id": plan.song_slug,
        "generated": datetime.now(timezone.utc).isoformat(),
        "source_arrangement": str(arrangement_path),
        "source_proposal": str(song_proposal_path) if song_proposal_path else None,
        # Identity — from song proposal (fallback to plan)
        "title": proposal_data.get("title", "") or plan.title,
        "bpm": proposal_data.get("bpm") or plan.bpm,
        "time_sig": proposal_data.get("time_sig", "") or plan.time_sig,
        "key": proposal_data.get("key", "") or plan.key,
        "rainbow_color": proposal_data.get("color", "") or plan.color,
        "singer": singer,
        "concept": proposal_data.get("concept", "") or plan.concept,
        "mood": proposal_data.get("mood") or plan.mood,
        "genres": proposal_data.get("genres") or plan.genres,
        "sounds_like": plan.sounds_like,
        # Track map (from --track-map arg)
        "tracks": track_map,
        # Clips (raw, from arrangement.txt)
        "clips": [
            {
                "name": c.name,
                "track": c.track,
                "start": c.start,
                "length": c.length,
            }
            for c in clips
        ],
        # Sections (derived from clips via derive_sections())
        "sections": [
            {
                "name": s.name,
                "start": s.start,
                "end": s.end,
                "vocals": s.vocals,
                **({"loops": s.loops} if s.loops else {}),
            }
            for s in sections
        ],
        # Render-time fields (null until produced)
        "vocals": vocals,
        "lyrics": lyrics,
        "release_date": None,
        "album_sequence": None,
        "main_audio_file": None,
        "TRT": None,
        "lrc_file": None,
        "audio_tracks": [],
    }

    out_path = production_dir / TRACK_MANIFEST_FILENAME
    with open(out_path, "w") as f:
        yaml.dump(
            data, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )
    return out_path


# ---------------------------------------------------------------------------
# MIDI assembly (flatten loops to events)
# ---------------------------------------------------------------------------


def _find_midi_file(name: str, production_dir: Path, instrument: str) -> Optional[Path]:
    """Find a MIDI by clip name in the instrument's approved/ then candidates/.

    Falls back to searching all instrument folders so candidate-ID-named clips
    (e.g. ``melody_verse_04``) are still found even when not promoted.
    """
    for subdir in ("approved", "candidates"):
        path = production_dir / instrument / subdir / f"{name}.mid"
        if path.exists():
            return path
    # Cross-folder fallback
    for instr in INSTRUMENT_FOLDERS:
        if instr == instrument:
            continue
        for subdir in ("approved", "candidates"):
            path = production_dir / instr / subdir / f"{name}.mid"
            if path.exists():
                return path
    return None


def assemble_midi_tracks(
    clips: list[Clip],
    production_dir: Path,
    output_dir: Path,
    bpm: float,
    track_map: Optional[dict[int, str]] = None,
    ticks_per_beat: int = 480,
) -> dict[int, Path]:
    """Assemble full-length MIDI files from arrangement clips.

    Reads each clip's MIDI from approved/ or candidates/, offsets all note
    events to the clip's absolute bar position, and writes one continuous
    MIDI per track.  No looping in the DAW is required — import the output
    files directly.

    Args:
        clips: Output of parse_arrangement().
        production_dir: Song production directory.
        output_dir: Directory to write assembled MIDIs (created if absent).
        bpm: Tempo used to convert seconds → ticks.
        track_map: {track_number: instrument_folder}.  Defaults to DEFAULT_TRACK_MAP.
        ticks_per_beat: MIDI resolution for output files.

    Returns:
        {track_number: output_path} for every track written.
    """
    if track_map is None:
        track_map = DEFAULT_TRACK_MAP

    output_dir.mkdir(parents=True, exist_ok=True)

    by_track: dict[int, list[Clip]] = {}
    for clip in clips:
        by_track.setdefault(clip.track, []).append(clip)

    written: dict[int, Path] = {}
    missing: list[str] = []

    for track_num, track_clips in sorted(by_track.items()):
        instrument = track_map.get(track_num, f"track_{track_num}")
        all_events: list[tuple[int, mido.Message]] = []

        for clip in sorted(track_clips, key=lambda c: c.start):
            midi_path = _find_midi_file(clip.name, production_dir, instrument)
            if midi_path is None:
                missing.append(f"{clip.name} (track {track_num}/{instrument})")
                continue

            base_tick = round(clip.start * bpm / 60.0 * ticks_per_beat)

            src = mido.MidiFile(filename=str(midi_path))
            scale = ticks_per_beat / src.ticks_per_beat

            # Strip Logic's baked-in timeline offset: Logic full-project exports
            # embed the region's absolute timeline position as leading silence.
            # Find the earliest note_on tick across all tracks and subtract it
            # so the MIDI is normalised to start at tick 0.
            min_src_tick: Optional[int] = None
            for track in src.tracks:
                abs_t = 0
                for msg in track:
                    abs_t += msg.time
                    if msg.type == "note_on" and msg.velocity > 0:
                        if min_src_tick is None or abs_t < min_src_tick:
                            min_src_tick = abs_t
                        break
            if min_src_tick is None:
                min_src_tick = 0

            for track in src.tracks:
                abs_src = 0
                for msg in track:
                    abs_src += msg.time
                    if msg.type in ("note_on", "note_off"):
                        dest_tick = base_tick + round((abs_src - min_src_tick) * scale)
                        all_events.append((dest_tick, msg.copy(time=0)))

        if not all_events:
            continue

        all_events.sort(key=lambda e: (e[0], e[1].type == "note_on"))

        out_mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        out_track = mido.MidiTrack()
        out_mid.tracks.append(out_track)
        out_track.append(
            mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0)
        )

        prev_tick = 0
        for abs_tick, msg in all_events:
            delta = abs_tick - prev_tick
            out_track.append(msg.copy(time=delta))
            prev_tick = abs_tick

        out_track.append(mido.MetaMessage("end_of_track", time=0))

        out_name = f"assembled_{instrument}.mid"
        out_path = output_dir / out_name
        out_mid.save(str(out_path))
        written[track_num] = out_path
        print(
            f"  Track {track_num} ({instrument}): "
            f"{len(track_clips)} clips → {out_path.name}"
        )

    for name in missing:
        print(f"  WARNING: MIDI file not found: {name}", file=sys.stderr)

    return written


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
    parser.add_argument(
        "--assemble",
        action="store_true",
        help=(
            "After importing, generate full-length assembled MIDI files "
            "(one per track) in <production-dir>/assembled/. "
            "Events are placed at absolute positions — no DAW looping required."
        ),
    )
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help=(
            "Skip plan/manifest update and only generate assembled MIDIs. "
            "Requires arrangement file and an existing production_plan.yml."
        ),
    )
    parser.add_argument(
        "--generate-manifest",
        action="store_true",
        help="Generate track_manifest.yml from arrangement.txt + song proposal.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help=(
            "Skip plan/drift update and only generate track_manifest.yml. "
            "Requires arrangement file and an existing production_plan.yml."
        ),
    )
    parser.add_argument(
        "--song-proposal",
        default=None,
        help="Path to song proposal YAML (auto-detected from production_plan if absent).",
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

    text = arr_path.read_text()
    proposal_path = Path(args.song_proposal) if args.song_proposal else None

    if args.manifest_only:
        print("=" * 60)
        print("TRACK MANIFEST GENERATION (manifest-only)")
        print("=" * 60)
        try:
            out = generate_track_manifest(
                prod_path,
                arr_path,
                song_proposal_path=proposal_path,
                track_map=track_map,
                vocalist_suffix=args.vocalist_suffix,
            )
            print(f"\nTrack manifest written: {out}")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        return

    if args.assemble_only:
        print("=" * 60)
        print("MIDI ASSEMBLY (assemble-only)")
        print("=" * 60)
        plan = load_plan(prod_path)
        if plan is None:
            print(f"ERROR: No {PLAN_FILENAME} found in {prod_path}")
            sys.exit(1)
        time_sig_parts = plan.time_sig.split("/")
        beats_per_bar = int(time_sig_parts[0]) * (4.0 / int(time_sig_parts[1]))
        clips = parse_arrangement(
            text, bpm=float(plan.bpm), beats_per_bar=beats_per_bar
        )
        out_dir = prod_path / "assembled"
        print(f"\nAssembling {len(clips)} clips → {out_dir}/")
        written = assemble_midi_tracks(
            clips, prod_path, out_dir, float(plan.bpm), track_map
        )
        print(f"\nWrote {len(written)} assembled MIDI tracks to {out_dir}/")
        return

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
    if any_drift:
        print(f"drift_report.yml written: {prod_path / DRIFT_REPORT_FILENAME}")

    if args.generate_manifest:
        try:
            out = generate_track_manifest(
                prod_path,
                arr_path,
                song_proposal_path=proposal_path,
                track_map=track_map,
                vocalist_suffix=args.vocalist_suffix,
            )
            print(f"track_manifest.yml written: {out}")
        except FileNotFoundError as e:
            print(f"WARNING: track manifest generation failed: {e}", file=sys.stderr)

    if args.assemble:
        plan = load_plan(prod_path)
        time_sig_parts = plan.time_sig.split("/")
        beats_per_bar = int(time_sig_parts[0]) * (4.0 / int(time_sig_parts[1]))
        clips = parse_arrangement(
            text, bpm=float(plan.bpm), beats_per_bar=beats_per_bar
        )
        out_dir = prod_path / "assembled"
        print(f"\nAssembling {len(clips)} clips → {out_dir}/")
        written = assemble_midi_tracks(
            clips, prod_path, out_dir, float(plan.bpm), track_map
        )
        print(f"Wrote {len(written)} assembled MIDI tracks")


if __name__ == "__main__":
    main()
