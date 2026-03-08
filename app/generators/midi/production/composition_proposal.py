#!/usr/bin/env python3
"""
Claude Composition Proposal — creative arrangement proposal before Logic session.

Reads the approved loop inventory across all pipeline phases, calls the Claude API
with full song context, and writes composition_proposal.yml to the production
directory. The proposal is advisory — no downstream pipeline depends on it — but
the drift report in assembly_manifest.py will compare it against the actual Logic
arrangement when both files exist.

Usage:
    python -m app.generators.midi.production.composition_proposal \
        --production-dir shrink_wrapped/.../production/blue__rust_signal_memorial_v1
"""

from __future__ import annotations

import argparse
import re
import sys
import mido
import yaml

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

PROPOSAL_FILENAME = "composition_proposal.yml"

# Instruments in pipeline order
INSTRUMENTS = ["chords", "drums", "bass", "melody"]

# ---------------------------------------------------------------------------
# Chromatic framework context for the prompt
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

The composition should serve the color's axes at every structural level: how sections
rise and fall, which loops carry weight, how the arrangement breathes. A Blue song
should feel like something Present but Forgotten — a place you can't quite locate,
memory eroding at the edges.
""".strip()


# ---------------------------------------------------------------------------
# Song proposal loading
# ---------------------------------------------------------------------------


def _find_chord_review(production_dir: Path) -> Optional[dict]:
    p = production_dir / "chords" / "review.yml"
    if not p.exists():
        return None
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _resolve_proposal_path(production_dir: Path, chord_review: dict) -> Optional[Path]:
    """Locate the song proposal YAML via chords/review.yml metadata."""
    thread = chord_review.get("thread", "")
    filename = chord_review.get("song_proposal", "")
    if not thread or not filename:
        return None
    for candidate in [
        Path(thread) / "yml" / filename,
        Path(thread) / filename,
    ]:
        if candidate.exists():
            return candidate
    return None


def load_song_proposal_data(production_dir: Path) -> dict:
    """Load the full song proposal including sounds_like and singer.

    Returns {} if the proposal cannot be found.
    """
    chord_review = _find_chord_review(production_dir)
    if not chord_review:
        return {}

    proposal_path = _resolve_proposal_path(production_dir, chord_review)
    if not proposal_path:
        return {}

    with open(proposal_path) as f:
        raw = yaml.safe_load(f) or {}

    color = raw.get("rainbow_color", {})
    if isinstance(color, dict):
        color = color.get("color_name", "")

    tempo = raw.get("tempo", {})
    if isinstance(tempo, dict):
        time_sig = f"{tempo.get('numerator', 4)}/{tempo.get('denominator', 4)}"
    else:
        time_sig = "4/4"

    return {
        "title": str(raw.get("title", "")),
        "bpm": int(raw.get("bpm", 120)),
        "time_sig": str(raw.get("time_sig") or time_sig),
        "key": str(raw.get("key", "")),
        "color": str(color) or str(chord_review.get("color", "")),
        "concept": str(raw.get("concept", "")),
        "mood": raw.get("mood") or [],
        "genres": raw.get("genres") or [],
        "sounds_like": raw.get("sounds_like") or [],
        "singer": str(raw.get("singer") or chord_review.get("singer", "")),
    }


# ---------------------------------------------------------------------------
# Loop inventory
# ---------------------------------------------------------------------------


def _midi_bar_count(midi_path: Path, bpm: int, time_sig: str) -> int:
    """Estimate bar count from a MIDI file."""
    try:
        mid = mido.MidiFile(str(midi_path))
        tpb = mid.ticks_per_beat or 480
        total_ticks = max(sum(msg.time for msg in track) for track in mid.tracks)
        beats_per_bar = int(time_sig.split("/")[0])
        ticks_per_bar = tpb * beats_per_bar
        return max(1, round(total_ticks / ticks_per_bar))
    except Exception:
        return 0


def build_loop_inventory(
    production_dir: Path, bpm: int = 120, time_sig: str = "4/4"
) -> dict:
    """Collect approved loops across all pipeline phases.

    Returns {instrument: [{label, bars, score, energy, notes}]}
    Missing or empty phases are omitted with a warning.
    """
    inventory: dict[str, list[dict]] = {}

    for instrument in INSTRUMENTS:
        review_path = production_dir / instrument / "review.yml"
        if not review_path.exists():
            continue
        with open(review_path) as f:
            review = yaml.safe_load(f) or {}

        approved = []
        for c in review.get("candidates", []):
            if c.get("status") != "approved":
                continue

            label = c.get("label") or c.get("section") or c.get("id", "")
            score = float(c.get("scores", {}).get("composite", 0.0))
            energy = str(c.get("energy", ""))
            notes = str(c.get("notes", ""))

            # Bar count — from approved MIDI if available
            midi_rel = c.get("midi_file", "")
            bars = 0
            if midi_rel:
                midi_path = production_dir / instrument / midi_rel
                if not midi_path.exists():
                    # Try approved/ subfolder
                    midi_path = (
                        production_dir / instrument / "approved" / Path(midi_rel).name
                    )
                if midi_path.exists():
                    bars = _midi_bar_count(midi_path, bpm, time_sig)

            approved.append(
                {
                    "label": label,
                    "bars": bars,
                    "score": round(score, 4),
                    "energy": energy,
                    "notes": notes,
                }
            )

        if approved:
            inventory[instrument] = approved

    return inventory


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _inventory_table(inventory: dict) -> str:
    """Format the loop inventory as a readable table."""
    lines = []
    for instrument, loops in inventory.items():
        lines.append(f"\n{instrument.upper()}:")
        for loop in loops:
            bar_str = f"{loop['bars']}b" if loop["bars"] else "?b"
            energy_str = f"  [{loop['energy']}]" if loop["energy"] else ""
            score_str = f"  score={loop['score']:.3f}"
            notes_str = f"  // {loop['notes']}" if loop["notes"] else ""
            lines.append(
                f"  {loop['label']:<35} {bar_str:<6}{score_str}{energy_str}{notes_str}"
            )
    return "\n".join(lines)


def build_prompt(proposal: dict, inventory: dict) -> str:
    """Build the system + user prompt for Claude."""
    color = proposal.get("color", "")
    sounds_like_str = ", ".join(proposal.get("sounds_like") or []) or "not specified"
    mood_str = ", ".join(proposal.get("mood") or []) or "not specified"
    genres_str = ", ".join(proposal.get("genres") or []) or "not specified"

    system = f"""{_CHROMATIC_CONTEXT}

You are a compositional collaborator on the White Project. Your job is to propose
a complete song arrangement before the human opens Logic Pro. Your proposal will be
compared against what they actually build — any divergence is interesting creative data.

Be specific and opinionated. Don't hedge. Make real artistic choices that serve the
concept and color target. The human will follow your proposal, deviate from it, or
use it as a foil — all three outcomes are valuable.

Return your response in two parts:
1. A fenced YAML block (```yaml ... ```) with the structured proposal
2. A "Rationale:" section with prose explaining your compositional reasoning

The YAML block MUST follow this exact schema. Use YAML block scalars (|) for
energy_note and transition_note — never bare strings with colons, which break parsing:
```yaml
sounds_like:
  - Artist Name
  - Artist Name
proposed_sections:
  - name: <loop_label>
    repeat: <integer>
    energy_note: |
      Brief energy/mood description. May contain colons freely.
    transition_note: |
      How this section moves into the next.
    loops:
      chords: <chord_loop_label or null>
      drums: <drum_loop_label or null>
      bass: <bass_loop_label or null>
      melody: <melody_loop_label or null>
```

Only use loop labels that exist in the inventory below. A section may omit
instruments that don't have a matching loop (set to null).
"""

    user = f"""Song: {proposal.get('title', 'Untitled')}
Color: {color}
Key: {proposal.get('key', '')}  BPM: {proposal.get('bpm', '')}  Time sig: {proposal.get('time_sig', '')}
Singer: {proposal.get('singer', '')}
Genres: {genres_str}
Mood: {mood_str}
Existing sounds_like: {sounds_like_str}

Concept:
{proposal.get('concept', '').strip()}

Available loops:
{_inventory_table(inventory)}

Propose a complete song arrangement using these loops. Decide how many times each
section repeats, the energy arc across the song, and how sections transition into
each other. Also propose your own sounds_like list — you may keep, extend, or
replace the existing one based on the arrangement you're imagining.
"""
    return system, user


# ---------------------------------------------------------------------------
# Claude API call
# ---------------------------------------------------------------------------


def call_claude(system: str, user: str, model: str = "claude-sonnet-4-6") -> str:
    """Call the Claude API and return the raw text response."""
    from anthropic import Anthropic

    client = Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_response(raw: str) -> tuple[dict, str]:
    """Extract structured YAML and rationale from Claude's response.

    Returns (structured_dict, rationale_prose).
    structured_dict is {} if no parseable YAML block found.
    """
    # Extract fenced YAML block
    match = re.search(r"```yaml\s*(.*?)```", raw, re.DOTALL)
    structured: dict = {}
    if match:
        try:
            structured = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            pass

    # Extract rationale prose (after "Rationale:" heading)
    rationale = ""
    rat_match = re.search(r"Rationale:\s*(.*)", raw, re.DOTALL | re.IGNORECASE)
    if rat_match:
        rationale = rat_match.group(1).strip()
    elif not structured:
        # Fallback: store entire response as rationale
        rationale = raw.strip()

    return structured, rationale


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_proposal(
    production_dir: Path,
    proposal_meta: dict,
    inventory: dict,
    structured: dict,
    rationale: str,
) -> Path:
    """Write composition_proposal.yml."""
    data = {
        "proposed_by": "claude",
        "generated": datetime.now(timezone.utc).isoformat(),
        "color_target": proposal_meta.get("color", ""),
        "title": proposal_meta.get("title", ""),
        "loop_inventory": inventory,
        "sounds_like": structured.get("sounds_like") or [],
        "proposed_sections": structured.get("proposed_sections") or [],
        "rationale": rationale,
    }
    out_path = production_dir / PROPOSAL_FILENAME
    with open(out_path, "w") as f:
        yaml.dump(
            data, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )
    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def generate(production_dir: Path, model: str = "claude-sonnet-4-6") -> Path:
    """Full pipeline: load → prompt → call → parse → write."""
    production_dir = Path(production_dir)

    print("Loading song proposal...")
    proposal = load_song_proposal_data(production_dir)
    if not proposal:
        print(
            "ERROR: Could not find song proposal. Ensure chords/review.yml has "
            "thread + song_proposal fields."
        )
        sys.exit(1)

    bpm = proposal.get("bpm", 120)
    time_sig = proposal.get("time_sig", "4/4")

    print("Building loop inventory...")
    inventory = build_loop_inventory(production_dir, bpm=bpm, time_sig=time_sig)
    if not inventory:
        print(
            "ERROR: No approved loops found — run pipeline phases before generating "
            "a composition proposal."
        )
        sys.exit(1)

    missing = [i for i in INSTRUMENTS if i not in inventory]
    if missing:
        print(
            f"Warning: no approved loops for: {', '.join(missing)} — proceeding with partial inventory"
        )

    total_loops = sum(len(v) for v in inventory.values())
    print(f"Found {total_loops} approved loops across {len(inventory)} instruments")

    print(f"Calling Claude ({model})...")
    system, user = build_prompt(proposal, inventory)
    raw = call_claude(system, user, model)

    structured, rationale = parse_response(raw)
    if not structured:
        print(
            "Warning: Could not parse structured proposal from Claude response — stored raw text"
        )
    else:
        n_sections = len(structured.get("proposed_sections") or [])
        n_sounds_like = len(structured.get("sounds_like") or [])
        print(
            f"  Parsed {n_sections} proposed sections, {n_sounds_like} sounds_like suggestions"
        )

    out_path = write_proposal(
        production_dir, proposal, inventory, structured, rationale
    )
    print(f"Written: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Claude composition proposal for a song production directory."
    )
    parser.add_argument(
        "--production-dir", required=True, help="Path to the song production directory"
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Claude model to use (default: claude-sonnet-4-6)",
    )
    args = parser.parse_args()
    generate(Path(args.production_dir), model=args.model)


if __name__ == "__main__":
    main()
