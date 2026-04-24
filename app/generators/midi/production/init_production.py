#!/usr/bin/env python3
"""
Production directory initialiser — generates an initial sounds_like list before
any MIDI pipeline phase runs.

Reads a song proposal YAML, asks Claude for 4–7 reference artists, and writes
`initial_proposal.yml` to the production directory. All downstream pipeline phases
(chords, drums, bass, melody, lyrics) read sounds_like from this file so artist
context is available from the very first generation step.

Usage:
    python -m app.generators.midi.production.init_production \
        --production-dir shrink_wrapped/.../production/red__my_song_v1 \
        --song-proposal shrink_wrapped/.../yml/song_proposal_Red_my_song_v1.yml
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

INITIAL_PROPOSAL_FILENAME = "initial_proposal.yml"
SONG_CONTEXT_FILENAME = "song_context.yml"

# Artists whose appearance in sounds_like indicates an ambient/shoegaze/drone aesthetic
_AMBIENT_CLUSTER: set[str] = {
    "grouper",
    "beach house",
    "my bloody valentine",
    "mbv",
    "low",
    "julianna barwick",
    "boards of canada",
    "stars of the lid",
    "arca",
    "william basinski",
    "the caretaker",
    "harold budd",
    "brian eno",
    "slowdive",
    "cocteau twins",
    "portishead",
    "mazzy star",
    "cigarettes after sex",
    "chelsea wolfe",
    "daughter",
}


def _detect_aesthetic_hints(sounds_like: list[str]) -> dict | None:
    """Return an aesthetic_hints dict if the sounds_like list implies a cluster.

    Returns None when no cluster is detected (no hints written to song_context).
    """
    if not sounds_like:
        return None
    lowered = {name.lower() for name in sounds_like}
    ambient_matches = lowered & _AMBIENT_CLUSTER
    if len(ambient_matches) >= 2:
        return {
            "density": "sparse",
            "texture": "hazy",
            "vocal_register": "lamentful",
        }
    return None


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def _build_sounds_like_prompt(meta: dict) -> str:
    """Build a Claude prompt requesting 4–7 reference artist names.

    meta keys used: concept, color, genres, mood.
    """
    color = meta.get("color", "")
    concept = meta.get("concept", "")
    genres = meta.get("genres") or []
    mood = meta.get("mood") or []

    genres_str = ", ".join(genres) if genres else "not specified"
    mood_str = (
        ", ".join(mood) if isinstance(mood, list) else str(mood or "not specified")
    )

    return f"""You are helping choose reference artists for a song.

Song details:
  Color: {color}
  Genres: {genres_str}
  Mood: {mood_str}
  Concept: {concept}

Return only a YAML list of 4–7 artist names that could serve as sonic references for this song.
No descriptions, no commentary, no parenthetical annotations — bare names only.

Example output:
- Sufjan Stevens
- Bon Iver
- The National
- Low
"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_sounds_like_response(text: str) -> list[str]:
    """Parse a bare artist name list from Claude's response.

    Handles YAML list format, numbered list, and one-per-line bare names.
    Strips parenthetical annotations if Claude adds them anyway.
    """
    names: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # Remove YAML list marker
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
        # Remove leading quotes
        stripped = stripped.strip("\"'")
        # Remove numbered list markers: "1. " or "1) "
        stripped = re.sub(r"^\d+[.)]\s+", "", stripped)
        # Strip parenthetical annotations: "Artist Name (context)" → "Artist Name"
        stripped = re.sub(r"\s*\([^)]*\)\s*$", "", stripped).strip()

        if stripped:
            names.append(stripped)

    return names


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def write_initial_proposal(
    production_dir: Path,
    meta: dict,
    sounds_like: list[str],
) -> Path:
    """Write song_context.yml to the production directory.

    The function name is preserved for call-site compatibility; initial_proposal.yml
    is no longer written. All pipeline phases read from song_context.yml which is a
    strict superset of the old initial_proposal.yml fields.

    Returns the path to song_context.yml.
    """
    production_dir = Path(production_dir)
    production_dir.mkdir(parents=True, exist_ok=True)
    return _write_song_context(production_dir, meta, sounds_like)


def _write_song_context(
    production_dir: Path,
    meta: dict,
    sounds_like: list[str],
    generated: str | None = None,
) -> Path:
    """Write song_context.yml — the canonical static metadata store.

    Returns the path to song_context.yml.
    """
    if generated is None:
        generated = datetime.now(timezone.utc).isoformat()

    aesthetic_hints = _detect_aesthetic_hints(sounds_like)

    # Style reference profile — extract from local MIDI files if available
    import logging as _logging

    from white_generation.style_reference import (  # circular import
        aggregate_profiles,
        load_or_extract_profile,
    )

    _style_profiles = []
    _missing_style_refs = []
    for _artist in sounds_like:
        _prof = load_or_extract_profile(_artist)
        if _prof is not None:
            _style_profiles.append(_prof)
        else:
            _missing_style_refs.append(_artist)

    if _missing_style_refs:
        _log = _logging.getLogger(__name__)
        if _style_profiles:
            _log.warning(
                "No style reference MIDI for %s — proceeding with partial coverage",
                ", ".join(repr(a) for a in _missing_style_refs),
            )
        else:
            _log.debug(
                "No style reference MIDI for any artist (%s) — skipping profile",
                ", ".join(repr(a) for a in _missing_style_refs),
            )
    _style_ref_profile = aggregate_profiles(_style_profiles)

    context_data = {
        "schema_version": "1",
        "generated": generated,
        "proposed_by": "claude",
        # Identity
        "title": meta.get("title", ""),
        "song_slug": meta.get("song_slug", ""),
        "song_proposal": meta.get("song_proposal", ""),
        "thread": meta.get("thread", ""),
        # Chromatic
        "color": meta.get("color", ""),
        "concept": meta.get("concept", ""),
        # Musical
        "key": meta.get("key", ""),
        "bpm": meta.get("bpm"),
        "time_sig": meta.get("time_sig", "4/4"),
        "singer": meta.get("singer", ""),
        # Creative context
        "sounds_like": sounds_like,
        "genres": meta.get("genres") or [],
        "mood": meta.get("mood") or [],
        # Aesthetic hints — written when sounds_like implies a recognised cluster
        **({"aesthetic_hints": aesthetic_hints} if aesthetic_hints else {}),
        # Style reference profile — averaged features from sounds_like MIDI files
        **(
            {"style_reference_profile": _style_ref_profile.model_dump()}
            if _style_ref_profile
            else {}
        ),
        # Phase status (updated by migration script)
        "phases": meta.get("phases")
        or {
            "chords": "pending",
            "drums": "pending",
            "bass": "pending",
            "melody": "pending",
            "lyrics": "pending",
            "composition_proposal": "pending",
        },
    }

    ctx_path = production_dir / SONG_CONTEXT_FILENAME
    with open(ctx_path, "w") as f:
        yaml.dump(
            context_data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    return ctx_path


def load_song_context(production_dir: Path) -> dict:
    """Load song_context.yml from the production directory.

    Returns {} if the file does not exist (graceful fallback for dirs that
    predate this change).
    """
    path = Path(production_dir) / SONG_CONTEXT_FILENAME
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_initial_proposal(production_dir: Path) -> dict:
    """Load song metadata from the production directory.

    Prefers song_context.yml (the canonical source of truth). Falls back to
    initial_proposal.yml only for legacy directories that predate this change.

    Returns {} if neither file exists.
    """
    ctx = load_song_context(production_dir)
    if ctx:
        return ctx
    # Fallback: legacy directory that only has initial_proposal.yml
    path = Path(production_dir) / INITIAL_PROPOSAL_FILENAME
    if path.exists():
        return yaml.safe_load(path.read_text()) or {}
    return {}


# ---------------------------------------------------------------------------
# Claude API call
# ---------------------------------------------------------------------------


_FALLBACK_SOUNDS_LIKE = [
    "boards of canada",
    "arca",
    "autechre",
    "burial",
    "four tet",
]


def _call_claude(prompt: str, model: str) -> str:
    from anthropic import Anthropic

    client = Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    if not response.content:
        if response.stop_reason == "refusal":
            print(
                "  Claude declined the sounds_like prompt (refusal) — using fallback artists."
            )
            return "\n".join(f"- {a}" for a in _FALLBACK_SOUNDS_LIKE)
        raise ValueError(
            f"Claude returned empty content (stop_reason={response.stop_reason!r})"
        )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def init_production(
    production_dir: Path,
    song_proposal_path: Path,
    model: str = "claude-sonnet-4-6",
    force: bool = False,
) -> Path:
    """Initialise a production directory by writing song_context.yml.

    If song_context.yml already exists and force=False, returns the existing
    path without regenerating.

    Returns the path to song_context.yml.
    """
    production_dir = Path(production_dir)
    out_path = production_dir / SONG_CONTEXT_FILENAME

    if out_path.exists() and not force:
        print(f"song_context.yml already exists at {out_path}")
        print("Use --force to regenerate.")
        return out_path

    # Load song proposal
    if not song_proposal_path.exists():
        print(f"ERROR: Song proposal not found: {song_proposal_path}")
        sys.exit(1)

    # Use production_plan loader (handles both nested tempo and flat time_sig formats)
    from app.generators.midi.production.production_plan import load_song_proposal

    meta = load_song_proposal(song_proposal_path)

    # Also grab singer from the raw proposal if available
    with open(song_proposal_path) as f:
        raw = yaml.safe_load(f) or {}
    meta["singer"] = str(raw.get("singer", ""))

    print(f"Song:    {meta.get('title', '(untitled)')}")
    print(f"Color:   {meta.get('color', '')}")
    print(f"Concept: {meta.get('concept', '')[:80]}")

    # Duplicate title check — walk all song_context.yml files in shrink_wrapped/
    _title = (meta.get("title") or "").strip().lower()
    if _title:
        _shrink_wrapped_root = production_dir.parent.parent.parent
        for _ctx_path in _shrink_wrapped_root.rglob("song_context.yml"):
            if _ctx_path.parent == production_dir:
                continue
            try:
                _existing = yaml.safe_load(_ctx_path.read_text()) or {}
            except Exception:
                continue
            _existing_title = (_existing.get("title") or "").strip().lower()
            if _existing_title == _title:
                print(
                    f"\nERROR: A production for '{meta.get('title')}' already exists at:\n"
                    f"  {_ctx_path.parent}\n"
                    "Use --force to initialise anyway."
                )
                if not force:
                    sys.exit(1)
                print("  --force passed — continuing despite duplicate.")
                break

    # Generate sounds_like via Claude
    print(f"\nAsking Claude ({model}) for reference artists...")
    prompt = _build_sounds_like_prompt(meta)
    raw_response = _call_claude(prompt, model)
    sounds_like = _parse_sounds_like_response(raw_response)

    if not sounds_like:
        print(
            "Warning: Claude returned no artist names — writing empty sounds_like list"
        )

    print(f"sounds_like ({len(sounds_like)}):")
    for name in sounds_like:
        print(f"  - {name}")

    ctx_path = write_initial_proposal(production_dir, meta, sounds_like)
    print(f"\nWritten: {ctx_path}")
    return ctx_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Initialise a production directory by generating an initial sounds_like list "
            "from the song proposal. Run this before chord_pipeline."
        )
    )
    parser.add_argument(
        "--production-dir",
        default=None,
        help=(
            "Song production directory (will be created if missing). "
            "Defaults to <thread>/production/<slug> derived from --song-proposal."
        ),
    )
    parser.add_argument(
        "--song-proposal",
        required=True,
        help="Path to the song proposal YAML file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate initial_proposal.yml even if it already exists",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Claude model to use (default: claude-sonnet-4-6)",
    )

    args = parser.parse_args()

    song_proposal_path = Path(args.song_proposal)
    if args.production_dir is None:
        from white_generation.pipelines.chord_pipeline import (
            song_slug,
        )  # circular import

        slug = song_slug(song_proposal_path.name)
        production_dir = song_proposal_path.parent.parent / "production" / slug
        print(f"Derived production dir: {production_dir}")
    else:
        production_dir = Path(args.production_dir)

    init_production(
        production_dir=production_dir,
        song_proposal_path=song_proposal_path,
        model=args.model,
        force=args.force,
    )


if __name__ == "__main__":
    main()
