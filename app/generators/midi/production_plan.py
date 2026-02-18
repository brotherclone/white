#!/usr/bin/env python3
"""
Production plan generator for the Music Production Pipeline.

Generates production_plan.yml — the structural backbone that maps approved
loops to song sections, defines the arrangement, and bridges toward a final
song manifest.

Usage:
    # Generate initial plan from approved chords
    python -m app.generators.midi.production_plan \
        --production-dir shrinkwrapped/.../production/black__sequential_dissolution_v2

    # Refresh bar counts from current approved loops (preserves human edits)
    python -m app.generators.midi.production_plan \
        --production-dir ... --refresh

    # Bootstrap a partial manifest from the completed plan
    python -m app.generators.midi.production_plan \
        --production-dir ... --bootstrap-manifest
"""

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import mido
import yaml

PLAN_FILENAME = "production_plan.yml"
MANIFEST_BOOTSTRAP_FILENAME = "manifest_bootstrap.yml"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PlanSection:
    name: str
    bars: int
    repeat: int = 1
    vocals: bool = False
    notes: str = ""
    _bar_source: str = field(default="", repr=False)  # derivation source (internal)


@dataclass
class ProductionPlan:
    song_slug: str
    generated: str
    bpm: int
    time_sig: str
    key: str
    color: str
    title: str = ""
    source_proposal: Optional[str] = None
    vocals_planned: bool = False
    sounds_like: list = field(default_factory=list)
    genres: list = field(default_factory=list)
    mood: list = field(default_factory=list)
    concept: str = ""
    sections: list = field(default_factory=list)  # list[PlanSection]


# ---------------------------------------------------------------------------
# Bar count derivation
# ---------------------------------------------------------------------------


def _midi_bar_count(midi_path: Path, bpm: int, time_sig: tuple[int, int]) -> int:
    """Compute bar count from a MIDI file using tick arithmetic."""
    try:
        mid = mido.MidiFile(str(midi_path))
        tpb = mid.ticks_per_beat or 480
        beats_per_bar = time_sig[0] * (4.0 / time_sig[1])
        bar_ticks = tpb * beats_per_bar

        max_tick = 0
        for track in mid.tracks:
            abs_tick = 0
            for msg in track:
                abs_tick += msg.time
            max_tick = max(max_tick, abs_tick)

        bars = round(max_tick / bar_ticks)
        return max(bars, 1)
    except Exception:
        return 0


def derive_bar_count(
    label: str,
    production_dir: Path,
    bpm: int,
    time_sig: tuple[int, int],
    chord_count_fallback: int = 4,
) -> tuple[int, str]:
    """Derive bar count for a section label.

    Priority:
    1. Approved harmonic rhythm MIDI (actual loop length post-rhythm)
    2. Approved chord MIDI
    3. chord_count_fallback (1 bar per chord)

    Returns (bars, source_description).
    """
    label_key = label.lower().replace("-", "_").replace(" ", "_")

    # 1. Harmonic rhythm approved
    hr_dir = production_dir / "harmonic_rhythm" / "approved"
    if hr_dir.exists():
        for pattern in (f"{label_key}*.mid", f"hr_{label_key}*.mid"):
            matches = sorted(hr_dir.glob(pattern))
            if matches:
                bars = _midi_bar_count(matches[0], bpm, time_sig)
                if bars > 0:
                    return bars, "harmonic_rhythm"

    # 2. Chord approved
    chord_dir = production_dir / "chords" / "approved"
    if chord_dir.exists():
        matches = sorted(chord_dir.glob(f"{label_key}*.mid"))
        if matches:
            bars = _midi_bar_count(matches[0], bpm, time_sig)
            if bars > 0:
                return bars, "chords"

    # 3. Fallback
    return chord_count_fallback, "chord_count"


# ---------------------------------------------------------------------------
# Plan I/O
# ---------------------------------------------------------------------------


def load_plan(production_dir: Path) -> Optional[ProductionPlan]:
    """Load existing production_plan.yml, or return None if absent."""
    plan_path = production_dir / PLAN_FILENAME
    if not plan_path.exists():
        return None
    with open(plan_path) as f:
        data = yaml.safe_load(f)

    sections = []
    for s in data.get("sections", []):
        sections.append(
            PlanSection(
                name=s["name"],
                bars=int(s["bars"]),
                repeat=int(s.get("repeat", 1)),
                vocals=bool(s.get("vocals", False)),
                notes=str(s.get("notes", "") or ""),
            )
        )

    return ProductionPlan(
        song_slug=data.get("song_slug", ""),
        generated=data.get("generated", ""),
        bpm=int(data.get("bpm", 120)),
        time_sig=str(data.get("time_sig", "4/4")),
        key=str(data.get("key", "")),
        color=str(data.get("color", "")),
        title=str(data.get("title", "")),
        source_proposal=data.get("source_proposal"),
        vocals_planned=bool(data.get("vocals_planned", False)),
        sounds_like=data.get("sounds_like") or [],
        genres=data.get("genres") or [],
        mood=data.get("mood") or [],
        concept=str(data.get("concept", "")),
        sections=sections,
    )


def save_plan(plan: ProductionPlan, production_dir: Path) -> Path:
    """Write production_plan.yml and return its path."""
    plan_path = production_dir / PLAN_FILENAME
    data = {
        "song_slug": plan.song_slug,
        "generated": plan.generated,
        "source_proposal": plan.source_proposal,
        "title": plan.title,
        "bpm": plan.bpm,
        "time_sig": plan.time_sig,
        "key": plan.key,
        "color": plan.color,
        "genres": plan.genres,
        "mood": plan.mood,
        "concept": plan.concept,
        "vocals_planned": plan.vocals_planned,
        "sounds_like": plan.sounds_like,
        "sections": [
            {
                "name": s.name,
                "bars": s.bars,
                "repeat": s.repeat,
                "vocals": s.vocals,
                "notes": s.notes,
            }
            for s in plan.sections
        ],
    }
    with open(plan_path, "w") as f:
        yaml.dump(
            data, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )
    return plan_path


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _read_chord_review(production_dir: Path) -> dict:
    review_path = production_dir / "chords" / "review.yml"
    if not review_path.exists():
        raise FileNotFoundError(f"Chord review not found: {review_path}")
    with open(review_path) as f:
        return yaml.safe_load(f)


def load_song_proposal(proposal_path: Path) -> dict:
    """Load a song proposal YAML and return normalised metadata dict.

    Handles both flat and nested tempo format (numerator/denominator dict
    vs. "4/4" string), and the rainbow_color dict structure.
    """
    with open(proposal_path) as f:
        raw = yaml.safe_load(f)

    # Time signature — stored as {numerator: N, denominator: D}
    tempo = raw.get("tempo", {})
    if isinstance(tempo, dict):
        time_sig = f"{tempo.get('numerator', 4)}/{tempo.get('denominator', 4)}"
    else:
        time_sig = str(tempo) if tempo else "4/4"

    # Rainbow color name
    color = raw.get("rainbow_color", {})
    if isinstance(color, dict):
        color = color.get("color_name", "")

    return {
        "title": str(raw.get("title", "")),
        "bpm": int(raw.get("bpm", 120)),
        "time_sig": time_sig,
        "key": str(raw.get("key", "")),
        "color": str(color),
        "genres": raw.get("genres") or [],
        "mood": raw.get("mood") or [],
        "concept": str(raw.get("concept", "")),
    }


def _parse_time_sig(time_sig_str: str) -> tuple[int, int]:
    parts = str(time_sig_str).split("/")
    return (int(parts[0]), int(parts[1]))


def generate_plan(
    production_dir: Path,
    proposal_path: Optional[Path] = None,
) -> ProductionPlan:
    """Generate a production_plan.yml from approved chord sections.

    Sections appear in the order they were labeled in the chord review.
    Bar counts are derived from approved MIDI files where available.

    If proposal_path is given, title, time_sig, genres, mood, and concept
    are read from the song proposal YAML (which is more authoritative than
    the chord review for these fields).
    """
    chord_review = _read_chord_review(production_dir)
    bpm = int(chord_review.get("bpm", 120))
    color = str(chord_review.get("color", ""))
    time_sig_str = str(chord_review.get("time_sig") or "4/4")
    key = str(chord_review.get("key", ""))
    title = str(chord_review.get("title") or "")
    genres: list = []
    mood: list = []
    concept: str = ""

    # Song proposal is authoritative for title, time_sig, genres, mood, concept
    proposal_data: dict = {}
    if proposal_path and proposal_path.exists():
        proposal_data = load_song_proposal(proposal_path)
        title = proposal_data.get("title", "") or title
        time_sig_str = proposal_data.get("time_sig", "") or time_sig_str
        key = proposal_data.get("key", "") or key
        color = proposal_data.get("color", "") or color
        bpm = proposal_data.get("bpm", bpm) or bpm
        genres = proposal_data.get("genres", [])
        mood = proposal_data.get("mood", [])
        concept = proposal_data.get("concept", "")

    time_sig = _parse_time_sig(time_sig_str)

    # Collect approved sections in order (first occurrence of each label)
    seen: set[str] = set()
    unique_sections = []
    for candidate in chord_review.get("candidates", []):
        status = str(candidate.get("status", "")).lower()
        if status not in ("approved", "accepted"):
            continue
        label = candidate.get("label")
        if not label:
            continue
        label_key = label.lower().replace("-", "_").replace(" ", "_")
        if label_key in seen:
            continue
        seen.add(label_key)
        chord_count = len(candidate.get("chords", [])) or 4
        unique_sections.append({"label": label, "chord_count": chord_count})

    if not unique_sections:
        raise ValueError("No approved chord sections found in chords/review.yml")

    sections = []
    for s in unique_sections:
        bars, source = derive_bar_count(
            s["label"], production_dir, bpm, time_sig, s["chord_count"]
        )
        sec = PlanSection(name=s["label"], bars=bars)
        sec._bar_source = source
        sections.append(sec)

    # Source proposal reference
    source_proposal = None
    thread = chord_review.get("thread", "")
    song_file = chord_review.get("song_proposal", "")
    if thread and song_file:
        source_proposal = str(Path(thread) / song_file)

    return ProductionPlan(
        song_slug=production_dir.name,
        generated=datetime.now(timezone.utc).isoformat(),
        bpm=bpm,
        time_sig=time_sig_str,
        key=key,
        color=color,
        title=title,
        source_proposal=source_proposal,
        genres=genres,
        mood=mood,
        concept=concept,
        sections=sections,
    )


def refresh_plan(production_dir: Path) -> ProductionPlan:
    """Reload bar counts from current approved loops, preserving human edits."""
    existing = load_plan(production_dir)
    if existing is None:
        raise FileNotFoundError(
            f"No {PLAN_FILENAME} found in {production_dir}. "
            "Run without --refresh to generate first."
        )

    chord_review = _read_chord_review(production_dir)
    time_sig = _parse_time_sig(existing.time_sig)

    # Current approved labels from chord review (for orphan detection)
    current_labels = {
        c["label"].lower().replace("-", "_").replace(" ", "_")
        for c in chord_review.get("candidates", [])
        if str(c.get("status", "")).lower() in ("approved", "accepted")
        and c.get("label")
    }

    refreshed = []
    for sec in existing.sections:
        label_key = sec.name.lower().replace("-", "_").replace(" ", "_")
        if label_key not in current_labels:
            print(
                f"  Warning: section '{sec.name}' not in current approved chords — retained"
            )
        bars, source = derive_bar_count(
            sec.name, production_dir, existing.bpm, time_sig, sec.bars
        )
        updated = PlanSection(
            name=sec.name,
            bars=bars,
            repeat=sec.repeat,
            vocals=sec.vocals,
            notes=sec.notes,
        )
        updated._bar_source = source
        refreshed.append(updated)

    existing.sections = refreshed
    existing.generated = datetime.now(timezone.utc).isoformat()
    return existing


# ---------------------------------------------------------------------------
# Manifest bootstrap
# ---------------------------------------------------------------------------


def bootstrap_manifest(production_dir: Path) -> Path:
    """Emit a partial manifest YAML from a completed production plan.

    All fields derivable from the plan are pre-filled. Fields that require
    a final render (audio files, TRT, timestamps) are written as null.
    """
    plan = load_plan(production_dir)
    if plan is None:
        raise FileNotFoundError(
            f"No {PLAN_FILENAME} found in {production_dir}. Generate a plan first."
        )

    time_sig_parts = plan.time_sig.split("/")
    beats_per_bar = int(time_sig_parts[0]) * (4.0 / int(time_sig_parts[1]))
    seconds_per_bar = beats_per_bar * (60.0 / plan.bpm)

    vocals_any = plan.vocals_planned or any(s.vocals for s in plan.sections)

    # Build structure with cumulative timestamps
    structure = []
    cursor = 0.0
    for sec in plan.sections:
        total_bars = sec.bars * sec.repeat
        duration = total_bars * seconds_per_bar
        end = cursor + duration

        def _fmt(t: float) -> str:
            m = int(t // 60)
            s = t % 60
            return f"[{m:02d}:{s:06.3f}]"

        structure.append(
            {
                "section_name": sec.name,
                "start_time": _fmt(cursor),
                "end_time": _fmt(end),
                "description": sec.notes or None,
            }
        )
        cursor = end

    data = {
        "manifest_id": plan.song_slug,
        "title": plan.title or plan.song_slug,
        "bpm": plan.bpm,
        "tempo": plan.time_sig,
        "key": plan.key,
        "rainbow_color": plan.color,
        "vocals": vocals_any,
        "lyrics": vocals_any,
        "sounds_like": plan.sounds_like,
        "mood": plan.mood,
        "genres": plan.genres,
        "concept": plan.concept,
        "structure": structure,
        # Render-time fields — fill in after final render
        "release_date": None,
        "album_sequence": None,
        "main_audio_file": None,
        "TRT": None,
        "lrc_file": None,
        "audio_tracks": [],
    }

    out_path = production_dir / MANIFEST_BOOTSTRAP_FILENAME
    with open(out_path, "w") as f:
        yaml.dump(
            data, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )
    return out_path


# ---------------------------------------------------------------------------
# next_section map helper (used by drum pipeline)
# ---------------------------------------------------------------------------


def build_next_section_map(plan: ProductionPlan) -> dict[str, Optional[str]]:
    """Build a mapping of section name → next section name from the plan.

    The last section maps to None. Used by the drum pipeline to annotate
    candidates with what section follows them.
    """
    if not plan.sections:
        return {}
    result = {}
    for i, sec in enumerate(plan.sections):
        next_sec = plan.sections[i + 1].name if i + 1 < len(plan.sections) else None
        result[sec.name] = next_sec
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Production plan — structural backbone for song arrangement"
    )
    parser.add_argument(
        "--production-dir", required=True, help="Song production directory"
    )
    parser.add_argument(
        "--song-proposal",
        default=None,
        help="Path to song proposal YAML — populates title, time_sig, genres, mood, concept",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh bar counts from current approved loops (preserves human edits)",
    )
    parser.add_argument(
        "--bootstrap-manifest",
        action="store_true",
        help="Emit partial manifest YAML from completed plan",
    )
    args = parser.parse_args()

    prod_path = Path(args.production_dir)
    if not prod_path.exists():
        print(f"ERROR: Production directory not found: {prod_path}")
        sys.exit(1)

    if args.bootstrap_manifest:
        try:
            out_path = bootstrap_manifest(prod_path)
            print(f"Manifest bootstrap written: {out_path}")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        return

    print("=" * 60)
    print("PRODUCTION PLAN GENERATOR")
    print("=" * 60)

    proposal_path = Path(args.song_proposal) if args.song_proposal else None
    if proposal_path and not proposal_path.exists():
        print(f"ERROR: Song proposal not found: {proposal_path}")
        sys.exit(1)

    try:
        if args.refresh:
            plan = refresh_plan(prod_path)
            # Apply proposal overrides on top of refreshed plan
            if proposal_path:
                proposal_data = load_song_proposal(proposal_path)
                plan.title = proposal_data.get("title", "") or plan.title
                plan.time_sig = proposal_data.get("time_sig", "") or plan.time_sig
                plan.key = proposal_data.get("key", "") or plan.key
                plan.genres = proposal_data.get("genres") or plan.genres
                plan.mood = proposal_data.get("mood") or plan.mood
                plan.concept = proposal_data.get("concept", "") or plan.concept
            print("Mode: refresh (bar counts updated, human edits preserved)")
        else:
            plan_path = prod_path / PLAN_FILENAME
            if plan_path.exists():
                print(
                    f"ERROR: {PLAN_FILENAME} already exists. Use --refresh to update."
                )
                sys.exit(1)
            plan = generate_plan(prod_path, proposal_path=proposal_path)
            print("Mode: generate")
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    out_path = save_plan(plan, prod_path)

    print(f"\nSong:    {plan.title or plan.song_slug}")
    print(f"BPM:     {plan.bpm}")
    print(f"Time:    {plan.time_sig}")
    print(f"Color:   {plan.color}")
    print(f"\nSections ({len(plan.sections)}):")
    for sec in plan.sections:
        source = f"[from {sec._bar_source}]" if sec._bar_source else ""
        print(
            f"  {sec.name:<15} {sec.bars} bars × {sec.repeat}"
            f"  vocals={sec.vocals}  {source}"
        )
    print(f"\nPlan written: {out_path}")
    print(f"Edit {PLAN_FILENAME} to set repeat counts, vocals, and section order")

    return plan


if __name__ == "__main__":
    main()
