#!/usr/bin/env python3
"""
Production plan generator for the Music Production Pipeline.

Generates production_plan.yml — the structural backbone that maps approved
loops to song sections, defines the arrangement, and bridges toward a final
song manifest.

Usage:
    # Generate initial plan from approved chords
    python -m app.generators.midi.production.production_plan \
        --production-dir shrink_wrapped/.../production/black__sequential_dissolution_v2

    # Refresh bar counts from current approved loops (preserves human edits)
    python -m app.generators.midi.production.production_plan \
        --production-dir ... --refresh

    # Bootstrap a partial manifest from the completed plan
    python -m app.generators.midi.production.production_plan \
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
from white_core.enums.lyric_repeat_type import LyricRepeatType

PLAN_FILENAME = "production_plan.yml"
MANIFEST_BOOTSTRAP_FILENAME = "manifest_bootstrap.yml"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PlanSection:
    name: str
    bars: int
    play_count: int = 1
    vocals: bool = True
    notes: str = ""
    reason: str = ""  # Claude's per-section compositional note
    lyric_repeat_type: str = "fresh"  # exact | variation | fresh
    arc: float = 0.0  # emotional intensity 0.0–1.0 (0=silence, 1=peak)
    loops: dict = field(default_factory=dict)  # {instrument: loop_name}
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
    vocals_planned: bool = True
    sounds_like: list = field(default_factory=list)
    genres: list = field(default_factory=list)
    mood: list = field(default_factory=list)
    concept: str = ""
    sections: list = field(default_factory=list)  # list[PlanSection]
    proposed_by: str = ""  # "claude" when arrangement was AI-proposed
    rationale: str = ""  # Claude's compositional reasoning


# ---------------------------------------------------------------------------
# Repeat type inference
# ---------------------------------------------------------------------------


def _normalize_repeat_type(value) -> LyricRepeatType:
    """Normalise a lyric_repeat_type value and validate.

    Accepts a LyricRepeatType, a string, or None. Returns the matching
    LyricRepeatType, falling back to FRESH on empty or unknown values so
    that a human typo (e.g. 'Exact') is silently corrected rather than
    silently breaking prompt generation.

    EXACT_REPEAT is an internal sentinel set programmatically — it must not
    be accepted from external YAML input and is treated as FRESH here.
    """
    if isinstance(value, LyricRepeatType):
        # Enum instances are always passed through as-is (including EXACT_REPEAT,
        # which is set programmatically, never from YAML)
        return value
    if not value:
        return LyricRepeatType.FRESH
    normalised = str(value).strip().lower()
    # EXACT_REPEAT is an internal sentinel — reject it as an external string value
    if normalised == LyricRepeatType.EXACT_REPEAT.value:
        return LyricRepeatType.FRESH
    try:
        return LyricRepeatType(normalised)
    except ValueError:
        return LyricRepeatType.FRESH


def _infer_repeat_type(label: str) -> LyricRepeatType:
    """Infer lyric_repeat_type from a section label.

    Rules (checked against the normalised lowercase label):
      - Contains 'pre_chorus' (checked before 'chorus') → VARIATION
      - Contains 'verse' → VARIATION
      - Contains 'chorus', 'refrain', or 'hook' → EXACT
      - Anything else → FRESH
    """
    norm = label.lower().replace("-", "_").replace(" ", "_")
    # pre_chorus must be checked before chorus (it contains the word 'chorus')
    if "pre_chorus" in norm:
        return LyricRepeatType.VARIATION
    if "verse" in norm:
        return LyricRepeatType.VARIATION
    if any(kw in norm for kw in ("chorus", "refrain", "hook")):
        return LyricRepeatType.EXACT
    return LyricRepeatType.FRESH


def _infer_arc_from_label(label: str) -> float:
    """Infer arc (emotional intensity 0.0–1.0) from a section label.

    Rules (checked against normalised lowercase label):
      - intro, outro → 0.15
      - bridge → 0.20
      - verse → 0.35
      - pre_chorus → 0.55
      - chorus, refrain, hook → 0.75
      - climax, peak → 0.90
      - anything else → 0.40
    """
    norm = label.lower().replace("-", "_").replace(" ", "_")
    if "pre_chorus" in norm:
        return 0.55
    if any(kw in norm for kw in ("intro", "outro")):
        return 0.15
    if "bridge" in norm:
        return 0.20
    if "verse" in norm:
        return 0.35
    if any(kw in norm for kw in ("chorus", "refrain", "hook")):
        return 0.75
    if any(kw in norm for kw in ("climax", "peak")):
        return 0.90
    return 0.40


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
    hr_distribution: Optional[list] = None,
) -> tuple[int, str]:
    """Derive bar count for a section label.

    Priority:
    1. hr_distribution from chord review.yml (sum of durations = total bars)
    2. Approved chord MIDI length
    3. chord_count_fallback (n chords × 1 bar default)

    Returns (bars, source_description).
    """
    label_key = label.lower().replace("-", "_").replace(" ", "_")

    # 1. hr_distribution field from chord review candidate
    if hr_distribution:
        bars = int(sum(float(d) for d in hr_distribution))
        if bars > 0:
            return bars, "hr_distribution"

    # 2. Chord approved MIDI
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
    """Load existing production_plan.yml, or return None if absent.

    Fields omitted from the YAML (bpm, time_sig, key, color, sounds_like) are
    filled from song_context.yml when present, then fall back to safe defaults.
    """
    plan_path = production_dir / PLAN_FILENAME
    if not plan_path.exists():
        return None
    with open(plan_path) as f:
        data = yaml.safe_load(f)

    # Fill fields that may be absent in plans written after 5.3 cleanup
    from app.generators.midi.production.init_production import load_song_context

    ctx = load_song_context(production_dir)

    sections = []
    for s in data.get("sections", []):
        sections.append(
            PlanSection(
                name=s["name"],
                bars=int(s["bars"]),
                play_count=int(s.get("play_count", s.get("repeat", 1))),
                vocals=bool(s.get("vocals", True)),
                notes=str(s.get("notes", "") or ""),
                reason=str(s.get("reason", "") or ""),
                lyric_repeat_type=_normalize_repeat_type(s.get("lyric_repeat_type")),
                loops=dict(s.get("loops") or {}),
                arc=float(s.get("arc", 0.0)),
            )
        )

    return ProductionPlan(
        song_slug=data.get("song_slug", ""),
        generated=data.get("generated", ""),
        bpm=int(data.get("bpm") or ctx.get("bpm") or 120),
        time_sig=str(data.get("time_sig") or ctx.get("time_sig") or "4/4"),
        key=str(data.get("key") or ctx.get("key") or ""),
        color=str(data.get("color") or ctx.get("color") or ""),
        title=str(data.get("title", "")),
        source_proposal=data.get("source_proposal"),
        vocals_planned=bool(data.get("vocals_planned", True)),
        sounds_like=data.get("sounds_like") or ctx.get("sounds_like") or [],
        genres=data.get("genres") or [],
        mood=data.get("mood") or [],
        concept=str(data.get("concept", "")),
        sections=sections,
        proposed_by=str(data.get("proposed_by") or ""),
        rationale=str(data.get("rationale") or ""),
    )


def save_plan(plan: ProductionPlan, production_dir: Path) -> Path:
    """Write production_plan.yml and return its path.

    Fields that duplicate song_context.yml (bpm, time_sig, key, color,
    sounds_like) are intentionally omitted; load_plan() pulls them from
    song_context.yml when present.
    """
    plan_path = production_dir / PLAN_FILENAME
    data = {
        "song_slug": plan.song_slug,
        "generated": plan.generated,
        **({"proposed_by": plan.proposed_by} if plan.proposed_by else {}),
        "source_proposal": plan.source_proposal,
        "title": plan.title,
        "genres": plan.genres,
        "mood": plan.mood,
        "concept": plan.concept,
        "vocals_planned": plan.vocals_planned,
        **({"rationale": plan.rationale} if plan.rationale else {}),
        "sections": [
            {
                "name": s.name,
                "bars": s.bars,
                "play_count": s.play_count,
                "vocals": s.vocals,
                "notes": s.notes,
                **({"reason": s.reason} if s.reason else {}),
                **(
                    {"lyric_repeat_type": _lrt.value}
                    if (_lrt := _normalize_repeat_type(s.lyric_repeat_type))
                    != LyricRepeatType.FRESH
                    else {}
                ),
                **({"loops": s.loops} if s.loops else {}),
                **({"arc": round(s.arc, 3)} if s.arc else {}),
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


def load_song_proposal_unified(
    proposal_path: Path,
    thread_dir: Optional[Path] = None,
) -> dict:
    """Canonical song proposal loader — single implementation for all pipeline phases.

    Supersedes the four divergent load_song_proposal implementations:
    - production_plan.load_song_proposal (this module)
    - chord_pipeline.load_song_proposal
    - lyric_pipeline._find_and_load_proposal (reads from chord_review)
    - composition_proposal.load_song_proposal_data

    Returns a normalised dict with all fields required by any pipeline phase:
        title, bpm, time_sig (always "N/N" string), key, key_root, mode,
        color, concept, genres, mood, singer, sounds_like, thread_dir, song_filename

    Args:
        proposal_path: Full path to the song proposal YAML file.
        thread_dir: Optional thread directory; if provided, manifest.yml is checked
                    for concept fallback (same as chord_pipeline behaviour).
    """
    with open(proposal_path) as f:
        raw = yaml.safe_load(f) or {}

    # Time signature — handles {numerator: N, denominator: D} dict or "4/4" string
    tempo = raw.get("tempo", {})
    if isinstance(tempo, dict):
        time_sig = f"{tempo.get('numerator', 4)}/{tempo.get('denominator', 4)}"
    else:
        time_sig = str(tempo) if tempo else "4/4"

    # Rainbow color
    color_raw = raw.get("rainbow_color", {})
    if isinstance(color_raw, dict):
        color = str(color_raw.get("color_name", ""))
    else:
        color = str(color_raw or "")

    # Key — return both combined string and parsed components
    key_str = str(raw.get("key", "C major"))
    key_root, mode = _parse_key_components(key_str)

    # Concept — song proposal first; manifest.yml fallback if thread_dir provided
    concept = str(raw.get("concept", ""))
    if not concept and thread_dir:
        manifest_path = Path(thread_dir) / "manifest.yml"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f) or {}
            concept = str(manifest.get("concept", ""))

    return {
        "title": str(raw.get("title", "")),
        "bpm": int(raw.get("bpm", 120)),
        "time_sig": time_sig,
        "key": key_str,
        "key_root": key_root,
        "mode": mode,
        "color": color,
        "concept": concept,
        "genres": raw.get("genres") or [],
        "mood": raw.get("mood") or [],
        "singer": str(raw.get("singer", "")),
        "sounds_like": raw.get("sounds_like") or [],
        "sub_proposals": [str(p) for p in (raw.get("sub_proposals") or [])],
        "thread_dir": str(thread_dir) if thread_dir else "",
        "song_filename": proposal_path.name,
        "raw_proposal": raw,
    }


def _parse_key_components(key_str: str) -> tuple[str, str]:
    """Parse a key string like 'F# minor' into (root, mode) components."""
    parts = key_str.strip().rsplit(" ", 1)
    if len(parts) == 2 and parts[1].lower() in ("minor", "major"):
        return parts[0], parts[1].capitalize()
    return key_str, "Major"


def _parse_time_sig(time_sig_str: str) -> tuple[int, int]:
    parts = str(time_sig_str).split("/")
    return (int(parts[0]), int(parts[1]))


def generate_plan_mechanical(
    production_dir: Path,
    proposal_path: Optional[Path] = None,
) -> ProductionPlan:
    """Generate a production_plan.yml from approved chord sections (mechanical).

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
        hr_dist = candidate.get("hr_distribution")
        unique_sections.append(
            {"label": label, "chord_count": chord_count, "hr_distribution": hr_dist}
        )

    if not unique_sections:
        raise ValueError("No approved chord sections found in chords/review.yml")

    # Section labels that conventionally carry no lead vocals
    _INSTRUMENTAL_SECTIONS = {
        "intro",
        "outro",
        "instrumental",
        "solo",
        "interlude",
        "break",
    }

    sections = []
    for s in unique_sections:
        bars, source = derive_bar_count(
            s["label"],
            production_dir,
            bpm,
            time_sig,
            s["chord_count"],
            hr_distribution=s.get("hr_distribution"),
        )
        label_key = s["label"].lower().replace("-", "_").replace(" ", "_")
        vocals = not any(
            label_key == v or label_key.startswith(v + "_")
            for v in _INSTRUMENTAL_SECTIONS
        )
        sec = PlanSection(
            name=s["label"],
            bars=bars,
            vocals=vocals,
            lyric_repeat_type=_infer_repeat_type(s["label"]),
            arc=_infer_arc_from_label(s["label"]),
        )
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


def generate_plan(
    production_dir: Path,
    proposal_path: Optional[Path] = None,
    use_claude: bool = True,
) -> ProductionPlan:
    """Generate a production plan, optionally with Claude arrangement proposal.

    Builds a mechanical inventory from approved chord sections, then — if
    use_claude=True (default) — calls propose_arrangement() to have Claude
    author a real arrangement arc with repeat counts, energy notes, and
    rationale.

    Falls back to the mechanical plan if the Anthropic API is unavailable or
    use_claude=False; prints a warning to stdout on fallback but does not raise.
    """
    plan = generate_plan_mechanical(production_dir, proposal_path=proposal_path)
    if use_claude:
        try:
            plan = propose_arrangement(plan)
        except Exception as e:
            print(
                f"  Warning: Claude arrangement unavailable ({e}) — using mechanical plan"
            )
    return plan


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
            play_count=sec.play_count,
            vocals=sec.vocals,
            notes=sec.notes,
            reason=sec.reason,
            lyric_repeat_type=sec.lyric_repeat_type,
            loops=dict(sec.loops),
            arc=sec.arc,  # preserve human override
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

    .. deprecated::
        Use ``assembly_manifest.generate_track_manifest()`` with ``--generate-manifest``
        instead. ``manifest_bootstrap.yml`` is derived from plan arithmetic; the new
        ``track_manifest.yml`` reads structure from the authoritative ``arrangement.txt``
        and identity from the song proposal YAML.  This function is retained for
        existing songs that already have a ``manifest_bootstrap.yml``.

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
        total_bars = sec.bars * sec.play_count
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
# Claude arrangement proposal
# ---------------------------------------------------------------------------

_CHROMATIC_CONTEXT = """
The White Project uses a chromatic synthesis framework where each rainbow color maps
to a set of axes:

  Red    — Past / Thing / Known
  Orange — Past / Place / Known
  Yellow — Present / Place / Imagined
  Green  — Present / Person / Imagined
  Blue   — Present / Place / Forgotten
  Indigo — Known, Forgotten (doesn't fit the standard axes — Indigo "isn't real")
  Violet — Future / Person / Imagined
  White  — synthesis of all colors

The arrangement should serve the color's axes at every structural level: how sections
rise and fall, how the song breathes, where tension accumulates and releases.
""".strip()


def _build_arrangement_prompt(plan: ProductionPlan) -> tuple[str, str]:
    """Build system + user prompt asking Claude to propose an arrangement arc."""
    mood_str = ", ".join(plan.mood) if plan.mood else "not specified"
    genres_str = ", ".join(plan.genres) if plan.genres else "not specified"

    system = f"""{_CHROMATIC_CONTEXT}

You are a compositional collaborator on the White Project. You have been given a set
of approved chord loop sections and must propose a complete song arrangement arc —
how many times each section repeats and in what order, and why.

Be specific and opinionated. Make real artistic choices that serve the concept and
the color's chromatic axes. The human will use your proposal as a starting point in
Logic Pro; they may follow it, diverge from it, or use it as a foil.

Return your response in two parts:
1. A fenced YAML block (```yaml ... ```) with the structured proposal
2. A "Rationale:" section with prose explaining your compositional reasoning

The YAML block MUST follow this exact schema:
```yaml
proposed_sections:
  - name: <section_name>
    play_count: <integer — total number of times this block plays>
    energy_note: <brief energy/mood description>
  - name: <section_name>
    play_count: <integer — total number of times this block plays>
    energy_note: <brief energy/mood description>
```

Only use section names from the available sections listed below. You may repeat
sections (e.g. Verse appearing multiple times in the list with different repeat counts
to create an A-B-A structure). The total runtime should feel like a complete song.
"""

    sections_table = "\n".join(
        f"  {s.name:<20} {s.bars} bars" + ("  [vocals]" if s.vocals else "")
        for s in plan.sections
    )

    user = f"""Song: {plan.title or plan.song_slug}
Color: {plan.color}
Key: {plan.key}  BPM: {plan.bpm}  Time sig: {plan.time_sig}
Genres: {genres_str}
Mood: {mood_str}

Concept:
{plan.concept.strip() if plan.concept else "(no concept provided)"}

Available sections:
{sections_table}

Propose a complete song arrangement using these sections. Set repeat counts,
decide if any sections should appear more than once in different positions, and
explain the energy arc you're imagining.
"""
    return system, user


def _parse_arrangement_response(raw: str, plan: ProductionPlan) -> tuple[list, str]:
    """Parse Claude's response into (updated_sections, rationale).

    Matches proposed section names back to the plan's existing PlanSection objects,
    updating repeat counts. Unrecognised names are skipped. If parsing fails entirely,
    the original sections are returned unchanged.
    """
    import re

    # Extract fenced YAML block
    match = re.search(r"```yaml\s*(.*?)```", raw, re.DOTALL)
    structured: dict = {}
    if match:
        try:
            structured = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            pass

    # Extract rationale
    rationale = ""
    rat_match = re.search(r"Rationale:\s*(.*)", raw, re.DOTALL | re.IGNORECASE)
    if rat_match:
        rationale = rat_match.group(1).strip()
    elif not structured:
        rationale = raw.strip()

    proposed = structured.get("proposed_sections") or []
    if not proposed:
        return list(plan.sections), rationale

    # Build lookup by normalised name
    section_by_key = {
        s.name.lower().replace("-", "_").replace(" ", "_"): s for s in plan.sections
    }

    updated: list[PlanSection] = []
    for entry in proposed:
        name = str(entry.get("name", "")).strip()
        key = name.lower().replace("-", "_").replace(" ", "_")
        original = section_by_key.get(key)
        if original is None:
            print(f"  Warning: proposed section '{name}' not in plan — skipped")
            continue
        play_count = int(entry.get("play_count", entry.get("repeat", 1)))
        energy_note = str(entry.get("energy_note", "") or "").strip()
        sec = PlanSection(
            name=original.name,
            bars=original.bars,
            play_count=max(1, play_count),
            vocals=original.vocals,
            notes=original.notes,
            reason=energy_note,
            loops=dict(original.loops),
            arc=original.arc,
        )
        updated.append(sec)

    if not updated:
        return list(plan.sections), rationale

    return updated, rationale


def propose_arrangement(plan: ProductionPlan) -> ProductionPlan:
    """Call Claude to propose an arrangement arc and update the plan in place.

    Sets plan.sections (with Claude's repeat counts), plan.rationale, and
    plan.proposed_by = "claude". Returns the modified plan.
    """
    from anthropic import Anthropic
    from dotenv import load_dotenv

    load_dotenv()

    system, user = _build_arrangement_prompt(plan)
    client = Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw = response.content[0].text

    updated_sections, rationale = _parse_arrangement_response(raw, plan)
    plan.sections = updated_sections
    plan.rationale = rationale
    plan.proposed_by = "claude"
    return plan


# ---------------------------------------------------------------------------
# Arrangement parsing (inlined from deleted assembly_manifest module)
# ---------------------------------------------------------------------------


@dataclass
class _Clip:
    start: float
    name: str
    track: int
    length: float


def _bars_beats_to_seconds(
    bar: int, beat: int, subdiv: int, tick: int, bpm: float, beats_per_bar: int
) -> float:
    total_beats = (bar - 1) * beats_per_bar + (beat - 1)
    frac_beats = (subdiv - 1) / 4.0 + (tick - 1) / (4.0 * 240.0)
    return (total_beats + frac_beats) * (60.0 / bpm)


def _is_bar_beat_format(text: str) -> bool:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        first_token = stripped.split()[0] if stripped.split() else ""
        return first_token.isdigit()
    return False


def _tc_to_seconds(tc: str) -> float:
    parts = tc.strip().split(":")
    try:
        if len(parts) >= 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2].split(".")[0])
        elif len(parts) == 2:
            h, m, s = 0, int(parts[0]), int(parts[1].split(".")[0])
        else:
            return 0.0
    except ValueError:
        return 0.0
    return h * 3600.0 + m * 60.0 + s


def _parse_bar_position(pos_str: str) -> tuple[int, int, int, int]:
    parts = pos_str.strip().split()
    return (
        int(parts[0]) if len(parts) > 0 else 1,
        int(parts[1]) if len(parts) > 1 else 1,
        int(parts[2]) if len(parts) > 2 else 1,
        int(parts[3]) if len(parts) > 3 else 1,
    )


def _parse_arrangement(
    text: str, bpm: float = 120.0, beats_per_bar: int = 4
) -> list[_Clip]:
    if _is_bar_beat_format(text):
        clips: list[_Clip] = []
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
            clips.append(
                _Clip(
                    start=round(start_secs - base_offset, 3),
                    name=name,
                    track=track,
                    length=round(end_secs - start_secs, 3),
                )
            )
        return clips

    # Timecode format
    clips = []
    base_offset = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            start_secs = _tc_to_seconds(parts[0])
            length_secs = _tc_to_seconds(parts[-1])
            track = int(parts[-2])
            name = " ".join(parts[1:-2])
        except (ValueError, IndexError):
            continue
        if base_offset is None:
            base_offset = start_secs
        clip_length = round(
            length_secs - start_secs if length_secs >= start_secs else length_secs, 3
        )
        clips.append(
            _Clip(
                start=round(start_secs - base_offset, 3),
                name=name,
                track=track,
                length=clip_length,
            )
        )
    return clips


# ---------------------------------------------------------------------------
# Sync plan from arrangement.txt
# ---------------------------------------------------------------------------


def parse_arrangement_sections(
    arrangement_path: Path,
    bpm: float = 120.0,
    beats_per_bar: int = 4,
) -> list[dict]:
    """Parse arrangement.txt and return per-instance section data.

    Handles both Logic Pro export formats (bar/beat and SMPTE timecode).
    Track 1 = chord/section clip (defines section identity and bar count).
    Track 4 = melody/vocals clip (presence means vocals=True for that instance).

    Returns list of dicts ordered by start time:
        {bar_start, section_name, bars, has_vocals}
    """
    text = arrangement_path.read_text()
    clips = _parse_arrangement(text, bpm=bpm, beats_per_bar=beats_per_bar)

    secs_per_bar = (60.0 / bpm) * beats_per_bar

    # Group clips by their start time (rounded to nearest 10ms to avoid float noise)
    by_start: dict[int, dict] = {}
    for clip in clips:
        key = round(clip.start * 100)  # centisecond bucket
        if key not in by_start:
            by_start[key] = {"start_secs": clip.start, "tracks": {}}
        bars = max(1, round(clip.length / secs_per_bar))
        by_start[key]["tracks"][clip.track] = {"name": clip.name, "bars": bars}

    instances = []
    for key in sorted(by_start):
        tracks = by_start[key]["tracks"]
        if 1 not in tracks:
            continue
        # Use 1-based bar number derived from start position
        bar_start = max(1, round(by_start[key]["start_secs"] / secs_per_bar) + 1)
        instances.append(
            {
                "bar_start": bar_start,
                "section_name": tracks[1]["name"],
                "bars": tracks[1]["bars"],
                "has_vocals": 4 in tracks,
            }
        )
    return instances


def sync_plan_from_arrangement(
    production_dir: Path,
    arrangement_path: Optional[Path] = None,
) -> ProductionPlan:
    """Rebuild plan sections from arrangement.txt.

    Expands grouped (play_count > 1) entries into individual play_count=1
    entries — one per section instance — and sets vocals per instance from
    track 4 (melody) presence in the arrangement.

    All other plan fields (rationale, concept, genres, etc.) are preserved.
    Per-section reason/notes/loops are carried forward by section name match.
    """
    existing = load_plan(production_dir)
    if existing is None:
        raise FileNotFoundError(
            f"No {PLAN_FILENAME} found in {production_dir}. Generate a plan first."
        )

    arr_path = arrangement_path or (production_dir / "arrangement.txt")
    if not arr_path.exists():
        raise FileNotFoundError(f"No arrangement.txt found at {arr_path}")

    _num, _den = (existing.time_sig or "4/4").split("/")
    _beats_per_bar = int(_num) * (4 // int(_den))
    instances = parse_arrangement_sections(
        arr_path, bpm=float(existing.bpm or 120), beats_per_bar=_beats_per_bar
    )
    if not instances:
        raise ValueError(f"No section instances parsed from {arr_path}")

    # Preserve per-section metadata from existing plan (first match by name)
    section_meta: dict[str, PlanSection] = {}
    for sec in existing.sections:
        key = sec.name.lower().replace("-", "_").replace(" ", "_")
        if key not in section_meta:
            section_meta[key] = sec

    new_sections = []
    for inst in instances:
        name = inst["section_name"]
        key = name.lower().replace("-", "_").replace(" ", "_")
        orig = section_meta.get(key)
        new_sections.append(
            PlanSection(
                name=name,
                bars=inst["bars"],
                play_count=1,
                vocals=inst["has_vocals"],
                notes=orig.notes if orig else "",
                reason=orig.reason if orig else "",
                lyric_repeat_type=(
                    orig.lyric_repeat_type if orig else _infer_repeat_type(name)
                ),
                loops=dict(orig.loops) if orig else {},
                arc=orig.arc if orig else _infer_arc_from_label(name),
            )
        )

    existing.sections = new_sections
    existing.generated = datetime.now(timezone.utc).isoformat()
    return existing


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
    parser.add_argument(
        "--no-claude",
        action="store_true",
        help="Skip Claude arrangement proposal and use the mechanical section inventory",
    )
    parser.add_argument(
        "--sync-from-arrangement",
        action="store_true",
        help=(
            "Rebuild sections from arrangement.txt — expands play_count groups into "
            "individual entries and sets vocals per instance from track 4 presence"
        ),
    )
    args = parser.parse_args()

    prod_path = Path(args.production_dir)
    if not prod_path.exists():
        print(f"ERROR: Production directory not found: {prod_path}")
        sys.exit(1)

    if args.sync_from_arrangement:
        try:
            plan = sync_plan_from_arrangement(prod_path)
            out_path = save_plan(plan, prod_path)
            print(f"Synced {len(plan.sections)} section instances from arrangement.txt")
            for sec in plan.sections:
                vocals_str = "vocals" if sec.vocals else "      "
                print(f"  {sec.name:<20} {sec.bars} bars  {vocals_str}")
            print(f"Plan written: {out_path}")
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        return

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

    use_claude = not args.no_claude
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
            plan = generate_plan(
                prod_path, proposal_path=proposal_path, use_claude=use_claude
            )
            print(f"Mode: generate ({'claude' if use_claude else 'mechanical'})")
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    out_path = save_plan(plan, prod_path)

    print(f"\nSong:    {plan.title or plan.song_slug}")
    print(f"BPM:     {plan.bpm}")
    print(f"Time:    {plan.time_sig}")
    print(f"Color:   {plan.color}")
    if plan.proposed_by:
        print(f"Proposed by: {plan.proposed_by}")
    print(f"\nSections ({len(plan.sections)}):")
    for sec in plan.sections:
        source = f"[from {sec._bar_source}]" if sec._bar_source else ""
        reason_str = f"  — {sec.reason}" if sec.reason else ""
        print(
            f"  {sec.name:<15} {sec.bars} bars × {sec.play_count}"
            f"  vocals={sec.vocals}  {source}{reason_str}"
        )
    if plan.rationale:
        print(
            f"\nRationale:\n{plan.rationale[:500]}{'...' if len(plan.rationale) > 500 else ''}"
        )
    print(f"\nPlan written: {out_path}")
    if not plan.proposed_by:
        print(f"Edit {PLAN_FILENAME} to set play_count, vocals, and section order")
    else:
        print(f"Edit {PLAN_FILENAME} to override Claude's arrangement choices")

    return plan


if __name__ == "__main__":
    main()
