#!/usr/bin/env python3
"""
Production decisions capture — structured ML training record for a completed song.

Reads the completed production directory and writes `production_decisions.yml`
containing the full decision chain: identity, per-phase candidate decisions,
arrangement structure, mix scores, and vocal drift.

Usage:
    python -m app.generators.midi.production.production_decisions \
        --production-dir shrink_wrapped/.../production/<slug>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import yaml

from white_composition.init_production import load_song_context

DECISIONS_FILENAME = "production_decisions.yml"

PHASES = ["chords", "drums", "bass", "melody"]


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def _load_identity(production_dir: Path) -> dict:
    """Read identity fields from song_context.yml."""
    ctx = load_song_context(production_dir)
    return {
        "thread": ctx.get("thread", ""),
        "color": ctx.get("color", ""),
        "title": ctx.get("title", ""),
        "key": ctx.get("key", ""),
        "bpm": ctx.get("bpm"),
        "time_sig": ctx.get("time_sig", "4/4"),
        "singer": ctx.get("singer", ""),
    }


# ---------------------------------------------------------------------------
# Phase decisions
# ---------------------------------------------------------------------------


def _load_phase_decisions(production_dir: Path, phase: str) -> Optional[dict]:
    """Read per-phase candidate and approval data from <phase>/review.yml.

    Returns None if the review file does not exist.
    """
    review_path = production_dir / phase / "review.yml"
    if not review_path.exists():
        return None

    with open(review_path) as f:
        data = yaml.safe_load(f) or {}

    candidates = data.get("candidates") or []
    approved = [
        c
        for c in candidates
        if str(c.get("status", "")).lower() in ("approved", "accepted")
    ]

    chromatic_scores = [
        c["scores"]["chromatic"]["match"]
        for c in approved
        if c.get("scores", {}).get("chromatic", {}).get("match") is not None
    ]
    theory_scores = [
        c["scores"]["theory_total"]
        for c in approved
        if c.get("scores", {}).get("theory_total") is not None
    ]

    mean_chromatic = (
        round(sum(chromatic_scores) / len(chromatic_scores), 4)
        if chromatic_scores
        else None
    )
    mean_theory = (
        round(sum(theory_scores) / len(theory_scores), 4) if theory_scores else None
    )

    return {
        "candidates_generated": len(candidates),
        "approved_count": len(approved),
        "approved_labels": [c.get("label") for c in approved if c.get("label")],
        "mean_chromatic_score": mean_chromatic,
        "mean_theory_score": mean_theory,
    }


# ---------------------------------------------------------------------------
# Arrangement summary
# ---------------------------------------------------------------------------


def _summarise_bar_beat(text: str) -> dict:
    """Parse a bar/beat format arrangement.txt for section structure.

    Track 1 clips define chord sections.  Track 4 clips define melody presence
    (vocals).  The length field's first number gives the bar count.
    """
    track1: list[tuple[str, int]] = []  # (label, bar_count)
    track4_names: set[str] = set()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        fields = [f.strip() for f in stripped.split("\t")]
        if len(fields) < 4:
            continue
        try:
            name = fields[1]
            track = int(fields[2])
            length_parts = fields[3].split()
            bar_count = int(length_parts[0]) if length_parts else 0
        except (ValueError, IndexError):
            continue

        if track == 1:
            track1.append((name, bar_count))
        elif track == 4:
            track4_names.add(name)

    # Aggregate per-label counts
    play_count: dict[str, int] = {}
    section_bars: dict[str, int] = {}
    for name, bars in track1:
        play_count[name] = play_count.get(name, 0) + 1
        section_bars[name] = bars  # last value wins (should be consistent)

    sections = []
    for label in play_count:
        has_vocals = any(label in t4 for t4 in track4_names)
        sections.append(
            {
                "name": label,
                "bars": section_bars[label],
                "play_count": play_count[label],
                "vocals": has_vocals,
            }
        )

    total_bars = sum(s["bars"] * s["play_count"] for s in sections)
    total_plays = sum(s["play_count"] for s in sections)

    return {
        "sections": sections,
        "total_bars": total_bars,
        "total_plays": total_plays,
        "section_count": len(sections),
    }


def _is_bar_beat_format(text: str) -> bool:
    """Return True if arrangement.txt uses bar/beat position format (e.g. '1 1 1 1\\tname\\t...')."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split("\t")
        if len(fields) >= 3:
            pos_parts = fields[0].split()
            if len(pos_parts) == 4 and all(p.isdigit() for p in pos_parts):
                return True
        return False
    return False


def _load_arrangement_summary(production_dir: Path) -> Optional[dict]:
    """Parse arrangement.txt for the arrangement summary block.

    Returns None if arrangement.txt does not exist.
    """
    arrangement_path = production_dir / "arrangement.txt"
    if not arrangement_path.exists():
        return None

    text = arrangement_path.read_text()
    return _summarise_bar_beat(text)


# ---------------------------------------------------------------------------
# Mix scores
# ---------------------------------------------------------------------------


def _load_mix_scores(production_dir: Path) -> Optional[dict]:
    """Read melody/mix_score.yml if present."""
    mix_path = production_dir / "melody" / "mix_score.yml"
    if not mix_path.exists():
        return None

    with open(mix_path) as f:
        data = yaml.safe_load(f) or {}

    result: dict = {}
    for dim in ("temporal", "spatial", "ontological"):
        dim_data = data.get(dim)
        if isinstance(dim_data, dict):
            # Store winning mode (highest value)
            winner = max(dim_data, key=lambda k: dim_data[k])
            result[dim] = {"scores": dim_data, "mode": winner}
        else:
            result[dim] = None

    result["confidence"] = data.get("confidence")
    result["chromatic_match"] = data.get("chromatic_match")
    return result


# ---------------------------------------------------------------------------
# Vocal drift
# ---------------------------------------------------------------------------


def _load_vocal_drift(production_dir: Path) -> Optional[dict]:
    """Read summary fields from drift_report.yml if present."""
    drift_path = production_dir / "drift_report.yml"
    if not drift_path.exists():
        return None

    with open(drift_path) as f:
        data = yaml.safe_load(f) or {}

    return {
        "overall_pitch_match": data.get("overall_pitch_match"),
        "overall_rhythm_drift": data.get("overall_rhythm_drift"),
        "total_lyric_edits": data.get("total_lyric_edits"),
    }


# ---------------------------------------------------------------------------
# Main assembler
# ---------------------------------------------------------------------------


def generate_decisions(production_dir: Path) -> dict:
    """Assemble the full production decisions record.

    Reads all available phase review files, arrangement, mix scores, and vocal
    drift.  Missing files produce None values rather than errors — partial
    productions are recorded faithfully.

    Returns a dict ready to be serialised to production_decisions.yml.
    """
    production_dir = Path(production_dir)

    identity = _load_identity(production_dir)

    phase_decisions: dict[str, Optional[dict]] = {}
    for phase in PHASES:
        phase_decisions[phase] = _load_phase_decisions(production_dir, phase)

    arrangement = _load_arrangement_summary(production_dir)
    mix_scores = _load_mix_scores(production_dir)
    vocal_drift = _load_vocal_drift(production_dir)

    return {
        "identity": identity,
        "phase_decisions": phase_decisions,
        "arrangement_summary": arrangement,
        "mix_scores": mix_scores,
        "vocal_drift": vocal_drift,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_decisions_file(production_dir: Path, decisions: dict) -> Path:
    """Write production_decisions.yml to the production directory root."""
    out_path = Path(production_dir) / DECISIONS_FILENAME
    with open(out_path, "w") as f:
        yaml.dump(
            decisions, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate production_decisions.yml for a completed song."
    )
    parser.add_argument("--production-dir", required=True)
    args = parser.parse_args()

    prod = Path(args.production_dir)
    if not prod.exists():
        print(f"ERROR: Production directory not found: {prod}")
        sys.exit(1)

    print(f"Generating production decisions for: {prod.name}")
    decisions = generate_decisions(prod)
    out_path = write_decisions_file(prod, decisions)

    identity = decisions["identity"]
    print(f"  Title  : {identity.get('title', '—')}")
    print(f"  Color  : {identity.get('color', '—')}")

    for phase, pd in decisions["phase_decisions"].items():
        if pd is None:
            print(f"  {phase:<8s}: no review.yml")
        else:
            print(
                f"  {phase:<8s}: {pd['approved_count']}/{pd['candidates_generated']} approved"
                f"  chromatic={pd['mean_chromatic_score']}"
            )

    arr = decisions["arrangement_summary"]
    if arr:
        print(
            f"  Arrangement: {arr['section_count']} sections, "
            f"{arr['total_bars']} bars, {arr['total_plays']} plays"
        )
    else:
        print("  Arrangement: no arrangement.txt")

    ms = decisions["mix_scores"]
    if ms:
        print(f"  Mix score  : chromatic_match={ms.get('chromatic_match')}")

    vd = decisions["vocal_drift"]
    if vd:
        print(f"  Vocal drift: pitch_match={vd.get('overall_pitch_match')}")

    print(f"  Written: {out_path}")


if __name__ == "__main__":
    main()
