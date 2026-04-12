#!/usr/bin/env python3
"""
Composition Narrative — Claude-generated multi-dimensional compositional brief.

Writes `composition_narrative.yml` to the production directory. The narrative
describes each section along four controlled-vocabulary dimensions (register,
texture, harmonic_complexity, rhythm_character), designates `the_moment`, and
provides free-text narrative for each section.

Usage:
    python -m app.generators.midi.production.composition_narrative \
        --production-dir shrink_wrapped/.../production/green__last_pollinators_elegy_v1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

from app.generators.midi.production.init_production import load_song_context
from app.generators.midi.production.production_plan import load_plan
from app.structures.music.narrative_constraints import (
    CompositionNarrative,
    extract_constraints,
)

load_dotenv()

log = logging.getLogger(__name__)

NARRATIVE_FILENAME = "composition_narrative.yml"

# Controlled vocabulary reference for the prompt
_VOCAB_REFERENCE = """
Controlled vocabularies (use ONLY these values):
  register:            low | low_mid | mid | mid_high | high
  texture:             absent | near_absent | sparse | moderate | full
  harmonic_complexity: simple | moderate | tense | rich
  rhythm_character:    absent | minimal | present | busy | open
  lead_voice:          bass | melody | none | drums | chords
""".strip()


# ---------------------------------------------------------------------------
# Claude API helpers
# ---------------------------------------------------------------------------


def _build_prompt(ctx: dict, plan_sections: list[dict]) -> str:
    """Compose the Claude prompt for narrative generation."""
    color = ctx.get("color", "Unknown")
    concept = ctx.get("concept", "")
    key = ctx.get("key", "")
    bpm = ctx.get("bpm", 120)
    time_sig = ctx.get("time_sig", "4/4")
    singer = ctx.get("singer", "")
    sounds_like = ctx.get("sounds_like") or []
    mood = ctx.get("mood") or []

    sounds_like_str = ", ".join(sounds_like) if sounds_like else "(none specified)"
    mood_str = ", ".join(mood) if mood else "(none specified)"
    sections_str = "\n".join(
        f"  {s['name']} ({s.get('bars', 4)} bars)" for s in plan_sections
    )

    return f"""You are a composer writing a detailed compositional brief for a song.

Song details:
  Color:       {color}
  Concept:     {concept}
  Key:         {key}
  BPM:         {bpm}
  Time sig:    {time_sig}
  Singer:      {singer or '(not specified)'}
  Sounds like: {sounds_like_str}
  Mood:        {mood_str}

Sections:
{sections_str}

Write a composition_narrative.yml for this song. Use YAML format. Include:

1. A `rationale` field: 2–4 sentences explaining the song's shape and where its
   emotional weight lives. Write as a composer thinking out loud.

2. A `the_moment` block with `section` (one section name) and `description` (1–2 sentences
   on why that section is the emotional peak and what should make it feel different).

3. A `sections` block with one entry per section. Each entry must have:
   - arc: float 0.0–1.0 (emotional intensity)
   - register, texture, harmonic_complexity, rhythm_character (from the controlled vocabularies)
   - lead_voice (which instrument carries the section)
   - narrative: one paragraph on what this section means and how it should feel

{_VOCAB_REFERENCE}

Write only valid YAML. Do not include any explanation outside the YAML block.
Start with `schema_version: "1"` and end with the last section entry.
"""


def generate_narrative(production_dir: str | Path) -> Optional[Path]:
    """Call Claude to generate composition_narrative.yml for a production directory.

    Returns the path to the written file, or None if generation failed.
    """
    try:
        import anthropic
    except ImportError:
        log.error("anthropic package not installed — cannot generate narrative")
        return None

    prod = Path(production_dir)
    ctx = load_song_context(prod)
    plan = load_plan(prod)

    if not ctx:
        log.warning("No song_context.yml found in %s — cannot generate narrative", prod)
        return None

    plan_sections: list[dict] = []
    if plan:
        plan_sections = [{"name": s.name, "bars": s.bars} for s in plan.sections]
    else:
        log.warning(
            "No production plan found — generating narrative with minimal section info"
        )

    prompt = _build_prompt(ctx, plan_sections)

    try:
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
    except Exception as exc:
        log.warning("Claude API call failed: %s", exc)
        return None

    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        start = next(
            (
                i
                for i, line in enumerate(lines)
                if line.strip().startswith("```yaml") or line.strip() == "```"
            ),
            0,
        )
        end = next(
            (i for i in range(len(lines) - 1, start, -1) if lines[i].strip() == "```"),
            len(lines),
        )
        raw = "\n".join(lines[start + 1 : end])

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        log.warning("Failed to parse Claude response as YAML: %s", exc)
        return None

    if not isinstance(data, dict):
        log.warning("Claude response is not a YAML dict — skipping write")
        return None

    out = prod / NARRATIVE_FILENAME
    out.write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    log.info("Narrative written: %s", out)
    return out


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_narrative(production_dir: str | Path) -> Optional[CompositionNarrative]:
    """Load composition_narrative.yml from a production directory.

    Returns None (with debug log) if the file is absent.
    """
    path = Path(production_dir) / NARRATIVE_FILENAME
    if not path.exists():
        log.debug("No composition_narrative.yml in %s", production_dir)
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return CompositionNarrative.from_dict(data or {})
    except Exception as exc:
        log.warning("Failed to load narrative: %s", exc)
        return None


def get_narrative_constraints(section_name: str, production_dir: str | Path) -> dict:
    """One-call helper: load narrative and extract constraints for a section.

    Returns {} if no narrative is present or the section is not in the narrative.
    """
    narrative = load_narrative(production_dir)
    if narrative is None:
        return {}
    return extract_constraints(section_name, narrative)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate composition_narrative.yml for a production directory"
    )
    parser.add_argument("--production-dir", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    result = generate_narrative(args.production_dir)
    if result is None:
        print("Narrative generation failed — check warnings above.")
        sys.exit(1)
    print(f"Narrative written: {result}")


if __name__ == "__main__":
    main()
