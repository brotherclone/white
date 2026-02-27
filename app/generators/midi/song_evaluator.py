#!/usr/bin/env python3
"""
Song Evaluator — AirGigs Readiness Metric

Aggregates chromatic scores and heuristic metrics from per-phase review.yml files
into a per-song song_evaluation.yml and a cross-song ranking CLI.

Usage:
    # Single song
    python -m app.generators.midi.song_evaluator <production_dir>

    # Cross-song comparison (rank multiple songs)
    python -m app.generators.midi.song_evaluator --compare <dir1> <dir2> ...

    # Optional: re-score promoted MIDIs with ChromaticScorer
    python -m app.generators.midi.song_evaluator <production_dir> --rescore
"""

import argparse
import glob as globmod
import re
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from app.generators.midi.production_plan import load_plan
from app.util.midi_cleanup import batch_trim as _midi_batch_trim

EVALUATION_FILENAME = "song_evaluation.yml"
COMPARISON_FILENAME = "comparison_report.yml"
PHASES = ("chords", "drums", "bass", "melody")

# Theory sub-fields by phase. None means use a top-level scores field directly.
_THEORY_FIELDS: dict[str, Optional[tuple]] = {
    "chords": ("melody", "voice_leading", "variety"),
    "drums": None,  # uses scores.energy_appropriateness directly
    "bass": ("root_adherence", "voice_leading", "kick_alignment"),
    "melody": ("singability", "chord_tone_alignment", "contour_quality"),
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PhaseReport:
    phase: str
    complete: bool
    sections_found: int
    sections_approved: int
    coverage: float
    mean_theory: float
    mean_chromatic_match: float
    mean_chromatic_confidence: float
    chromatic_consistency: float
    approved_labels: list


@dataclass
class EvaluationReport:
    song_slug: str
    color: str
    title: str
    evaluated: str

    phases: dict
    phases_complete: int

    # Arrangement
    total_bars: int
    unique_sections: int
    section_variety: float
    vocal_coverage: float
    has_lyrics: bool
    lyric_syllable_density: float

    # Structural integrity (from drift_report.yml; 1.0 if absent)
    max_drift_seconds: float
    name_mismatches: int
    structural_integrity: float

    # Heuristic scores (all 0.0–1.0)
    chromatic_alignment: float
    theory_quality: float
    production_completeness: float
    lyric_maturity: float

    # Composite + readiness
    composite_score: float
    airgigs_readiness: str
    flags: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Theory score helpers
# ---------------------------------------------------------------------------


def _compute_theory_score(candidate: dict, phase: str) -> float:
    """Extract and average theory sub-scores for a candidate."""
    scores = candidate.get("scores", {}) or {}
    if phase == "drums":
        return float(scores.get("energy_appropriateness", 0.0))
    theory = scores.get("theory", {}) or {}
    fields = _THEORY_FIELDS.get(phase) or ()
    if not fields:
        return 0.0
    vals = [float(theory.get(f, 0.0)) for f in fields]
    return sum(vals) / len(vals) if vals else 0.0


def _safe_mean(vals: list) -> float:
    """Mean of a list; returns 0.0 for empty."""
    return statistics.mean(vals) if vals else 0.0


def _chromatic_consistency(match_scores: list) -> float:
    """1.0 - stdev(match_scores). Returns 1.0 for fewer than 2 scores."""
    if len(match_scores) < 2:
        return 1.0
    try:
        return max(0.0, 1.0 - statistics.stdev(match_scores))
    except statistics.StatisticsError:
        return 1.0


# ---------------------------------------------------------------------------
# Phase report
# ---------------------------------------------------------------------------


def _load_phase_report(production_dir: Path, phase: str) -> Optional[PhaseReport]:
    """Load PhaseReport from {phase}/review.yml. Returns None if absent."""
    review_path = production_dir / phase / "review.yml"
    if not review_path.exists():
        return None

    with open(review_path) as f:
        data = yaml.safe_load(f)

    candidates = data.get("candidates", []) or []

    # sections_found: chords derive from non-null labels; others use explicit list
    if phase == "chords":
        labeled = [c for c in candidates if c.get("label") is not None]
        sections_found = len({c["label"] for c in labeled})
    else:
        found_list = data.get("sections_found", []) or []
        sections_found = len(found_list)

    # Approved candidates with non-null labels
    approved = [
        c
        for c in candidates
        if str(c.get("status", "")).lower() in ("approved", "accepted")
        and c.get("label") is not None
    ]

    # Unique sections with approved candidates
    if phase == "chords":
        approved_sections = {c["label"] for c in approved}
    else:
        approved_sections = {c.get("section", "") for c in approved if c.get("section")}
    sections_approved = len(approved_sections)

    coverage = sections_approved / sections_found if sections_found > 0 else 0.0
    approved_labels = [c["label"] for c in approved]

    # Theory scores (per approved candidate)
    theory_scores = [_compute_theory_score(c, phase) for c in approved]
    mean_theory = _safe_mean(theory_scores)

    # Chromatic scores
    match_scores: list = []
    confidence_scores: list = []
    for c in approved:
        ch = (c.get("scores", {}) or {}).get("chromatic", {}) or {}
        match_scores.append(float(ch.get("match", 0.0)))
        confidence_scores.append(float(ch.get("confidence", 0.0)))

    mean_chromatic_match = _safe_mean(match_scores)
    mean_chromatic_confidence = _safe_mean(confidence_scores)
    chromatic_consistency = _chromatic_consistency(match_scores)

    return PhaseReport(
        phase=phase,
        complete=sections_found > 0 and sections_approved > 0,
        sections_found=sections_found,
        sections_approved=sections_approved,
        coverage=coverage,
        mean_theory=mean_theory,
        mean_chromatic_match=mean_chromatic_match,
        mean_chromatic_confidence=mean_chromatic_confidence,
        chromatic_consistency=chromatic_consistency,
        approved_labels=approved_labels,
    )


# ---------------------------------------------------------------------------
# Syllable counting
# ---------------------------------------------------------------------------


def _count_syllables(line: str) -> int:
    """Estimate syllable count for a line of lyrics via vowel-group heuristic."""
    total = 0
    for word in re.findall(r"[a-zA-Z'-]+", line):
        for part in word.replace("'", "").split("-"):
            if not part:
                continue
            vowel_groups = len(re.findall(r"[aeiou]+", part.lower()))
            # Silent-e: trailing 'e' after consonant, but '-le' forms its own syllable
            if (
                part.lower().endswith("e")
                and len(part) > 2
                and part.lower()[-2] not in "aeiou"
                and not part.lower().endswith("le")
            ):
                vowel_groups = max(1, vowel_groups - 1)
            total += max(1, vowel_groups)
    return total


# ---------------------------------------------------------------------------
# Arrangement metrics
# ---------------------------------------------------------------------------


def _compute_arrangement_metrics(production_dir: Path) -> dict:
    """Read production_plan.yml and compute arrangement-level metrics."""
    plan = load_plan(production_dir)
    if plan is None or not plan.sections:
        return {
            "total_bars": 0,
            "unique_sections": 0,
            "section_variety": 0.0,
            "vocal_coverage": 0.0,
            "vocal_bars": 0,
        }

    total_bars = sum(s.bars * s.repeat for s in plan.sections)
    vocal_bars = sum(s.bars * s.repeat for s in plan.sections if s.vocals)
    all_names = [s.name for s in plan.sections]
    unique_names = set(all_names)
    unique_sections = len(unique_names)
    section_variety = unique_sections / len(all_names) if all_names else 0.0
    vocal_coverage = vocal_bars / total_bars if total_bars > 0 else 0.0

    return {
        "total_bars": total_bars,
        "unique_sections": unique_sections,
        "section_variety": section_variety,
        "vocal_coverage": vocal_coverage,
        "vocal_bars": vocal_bars,
    }


# ---------------------------------------------------------------------------
# Lyric metrics
# ---------------------------------------------------------------------------


def _compute_lyric_metrics(production_dir: Path, vocal_bars: int) -> tuple:
    """Return (has_lyrics, syllables_per_vocal_bar) from melody/lyrics.txt."""
    lyrics_path = production_dir / "melody" / "lyrics.txt"
    if not lyrics_path.exists():
        return False, 0.0

    text = lyrics_path.read_text(encoding="utf-8")
    lyric_lines = [
        line
        for line in text.splitlines()
        if line.strip()
        and not line.strip().startswith("#")
        and not line.strip().startswith("[")
    ]

    total_syllables = sum(_count_syllables(line) for line in lyric_lines)
    density = total_syllables / vocal_bars if vocal_bars > 0 else 0.0
    return True, density


# ---------------------------------------------------------------------------
# Structural integrity
# ---------------------------------------------------------------------------


def _compute_structural_integrity(production_dir: Path) -> tuple:
    """Return (structural_integrity, max_drift_seconds, name_mismatches).

    Returns (1.0, 0.0, 0) if drift_report.yml is absent.
    """
    drift_path = production_dir / "drift_report.yml"
    if not drift_path.exists():
        return 1.0, 0.0, 0

    with open(drift_path) as f:
        data = yaml.safe_load(f)

    sections = data.get("sections", []) or []
    if not sections:
        return 1.0, 0.0, 0

    max_drift = max(abs(float(s.get("drift_seconds", 0.0))) for s in sections)
    name_mismatches = sum(1 for s in sections if s.get("name_mismatch", False))
    total = len(sections)

    # Section name match rate is the primary structural indicator.
    # Timing drift reflects creative decisions (shorter arrangement than planned)
    # and is informational only — penalise lightly with a 120s ceiling so a
    # deliberately compact arrangement does not tank the score.
    mismatch_score = max(0.0, 1.0 - name_mismatches / total) if total > 0 else 1.0
    drift_score = max(0.0, 1.0 - max_drift / 120.0)
    structural_integrity = mismatch_score * 0.8 + drift_score * 0.2

    return structural_integrity, max_drift, name_mismatches


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------


def _compute_composite(
    chromatic_alignment: float,
    theory_quality: float,
    production_completeness: float,
    structural_integrity: float,
    lyric_maturity: float,
) -> float:
    """Weighted composite score (weights sum to 1.0)."""
    return (
        0.40 * chromatic_alignment
        + 0.25 * theory_quality
        + 0.20 * production_completeness
        + 0.10 * structural_integrity
        + 0.05 * lyric_maturity
    )


def _determine_readiness(composite: float) -> str:
    """Map composite score to AirGigs readiness tier."""
    if composite >= 0.75:
        return "ready"
    if composite >= 0.55:
        return "demo"
    return "draft"


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


def _collect_flags(report: "EvaluationReport") -> list:
    """Collect diagnostic flags for a report."""
    flags = []

    for phase in PHASES:
        if phase not in report.phases:
            flags.append(f"incomplete: missing {phase}")

    if report.has_lyrics and report.lyric_syllable_density < 0.5:
        flags.append("sparse lyrics")

    if report.max_drift_seconds > 2.0:
        flags.append(f"timing drift {report.max_drift_seconds:.1f}s")

    if report.name_mismatches > 0:
        flags.append(f"arrangement mismatch ({report.name_mismatches} sections)")

    # Confidence check across available phases
    all_confidence = [
        pr.mean_chromatic_confidence
        for pr in report.phases.values()
        if isinstance(pr, PhaseReport)
    ]
    if all_confidence and _safe_mean(all_confidence) < 0.15:
        flags.append("low chromatic confidence")

    if report.chromatic_alignment > 0.80:
        flags.append("strong chromatic alignment")

    if report.theory_quality > 0.80:
        flags.append("high theory quality")

    return flags


# ---------------------------------------------------------------------------
# Main evaluate function
# ---------------------------------------------------------------------------


def evaluate(production_dir: Path) -> EvaluationReport:
    """Evaluate a song production directory and write song_evaluation.yml."""
    production_dir = Path(production_dir)

    # Pre-flight: trim Logic-exported tempo track bloat from all approved MIDIs
    for _phase in PHASES:
        _approved = production_dir / _phase / "approved"
        if _approved.exists():
            _midi_batch_trim(_approved)

    plan = load_plan(production_dir)
    song_slug = production_dir.name
    color = plan.color if plan else ""
    title = plan.title if plan else ""

    # Phase reports
    phases: dict = {}
    for phase in PHASES:
        pr = _load_phase_report(production_dir, phase)
        if pr is not None:
            phases[phase] = pr

    phases_complete = sum(1 for pr in phases.values() if pr.complete)

    # Arrangement
    arr = _compute_arrangement_metrics(production_dir)
    total_bars = arr["total_bars"]
    unique_sections = arr["unique_sections"]
    section_variety = arr["section_variety"]
    vocal_coverage = arr["vocal_coverage"]
    vocal_bars = arr["vocal_bars"]

    # Lyrics
    has_lyrics, lyric_density = _compute_lyric_metrics(production_dir, vocal_bars)

    # Structural integrity
    structural_integrity, max_drift, name_mismatches = _compute_structural_integrity(
        production_dir
    )

    # Heuristic scores
    completed_phases = [pr for pr in phases.values() if pr.complete]
    chromatic_alignment = _safe_mean(
        [pr.mean_chromatic_match for pr in completed_phases]
    )
    theory_quality = _safe_mean([pr.mean_theory for pr in completed_phases])
    production_completeness = phases_complete / len(PHASES)
    lyric_maturity = 0.0 if not has_lyrics else min(1.0, lyric_density)

    composite_score = _compute_composite(
        chromatic_alignment,
        theory_quality,
        production_completeness,
        structural_integrity,
        lyric_maturity,
    )
    airgigs_readiness = _determine_readiness(composite_score)

    report = EvaluationReport(
        song_slug=song_slug,
        color=color,
        title=title,
        evaluated=datetime.now(timezone.utc).isoformat(),
        phases=phases,
        phases_complete=phases_complete,
        total_bars=total_bars,
        unique_sections=unique_sections,
        section_variety=section_variety,
        vocal_coverage=vocal_coverage,
        has_lyrics=has_lyrics,
        lyric_syllable_density=lyric_density,
        max_drift_seconds=max_drift,
        name_mismatches=name_mismatches,
        structural_integrity=structural_integrity,
        chromatic_alignment=chromatic_alignment,
        theory_quality=theory_quality,
        production_completeness=production_completeness,
        lyric_maturity=lyric_maturity,
        composite_score=composite_score,
        airgigs_readiness=airgigs_readiness,
        flags=[],
    )
    report.flags = _collect_flags(report)

    _write_evaluation(report, production_dir)
    return report


def _write_evaluation(report: EvaluationReport, production_dir: Path) -> Path:
    """Serialize EvaluationReport to song_evaluation.yml."""
    phases_data = {}
    for phase, pr in report.phases.items():
        phases_data[phase] = {
            "complete": pr.complete,
            "sections_found": pr.sections_found,
            "sections_approved": pr.sections_approved,
            "coverage": round(pr.coverage, 4),
            "mean_theory": round(pr.mean_theory, 4),
            "mean_chromatic_match": round(pr.mean_chromatic_match, 4),
            "mean_chromatic_confidence": round(pr.mean_chromatic_confidence, 4),
            "chromatic_consistency": round(pr.chromatic_consistency, 4),
            "approved_labels": pr.approved_labels,
        }

    data = {
        "song_slug": report.song_slug,
        "color": report.color,
        "title": report.title,
        "evaluated": report.evaluated,
        "phases_complete": report.phases_complete,
        "phases": phases_data,
        "total_bars": report.total_bars,
        "unique_sections": report.unique_sections,
        "section_variety": round(report.section_variety, 4),
        "vocal_coverage": round(report.vocal_coverage, 4),
        "has_lyrics": report.has_lyrics,
        "lyric_syllable_density": round(report.lyric_syllable_density, 4),
        "max_drift_seconds": round(report.max_drift_seconds, 2),
        "name_mismatches": report.name_mismatches,
        "structural_integrity": round(report.structural_integrity, 4),
        "chromatic_alignment": round(report.chromatic_alignment, 4),
        "theory_quality": round(report.theory_quality, 4),
        "production_completeness": round(report.production_completeness, 4),
        "lyric_maturity": round(report.lyric_maturity, 4),
        "composite_score": round(report.composite_score, 4),
        "airgigs_readiness": report.airgigs_readiness,
        "flags": report.flags,
    }

    out_path = production_dir / EVALUATION_FILENAME
    with open(out_path, "w") as f:
        yaml.dump(
            data, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )
    return out_path


# ---------------------------------------------------------------------------
# Rescore lyrics (optional --rescore-lyrics flag)
# ---------------------------------------------------------------------------


def _rescore_lyrics(production_dir: Path, plan_concept: str, plan_color: str) -> dict:
    """Score melody/lyrics.txt and melody/lyrics_draft.txt via ChromaticScorer text-only.

    Returns a dict with some subset of:
        lyrics_edited_chromatic_match, lyrics_draft_chromatic_match, lyrics_chromatic_delta

    Missing files are handled gracefully (empty dict or partial result).
    Exits with an error message if ChromaticScorer is unavailable.
    """
    from app.generators.midi.chord_pipeline import (
        compute_chromatic_match,
        get_chromatic_target,
    )

    melody_dir = production_dir / "melody"
    lyrics_path = melody_dir / "lyrics.txt"
    draft_path = melody_dir / "lyrics_draft.txt"

    if not lyrics_path.exists():
        print("WARNING: melody/lyrics.txt not found — skipping lyric rescore")
        return {}

    onnx_path = (
        Path(__file__).parent.parent.parent.parent
        / "training"
        / "data"
        / "fusion_model.onnx"
    )
    if not onnx_path.exists():
        print("ERROR: ONNX model not found.")
        print("Ensure training/data/fusion_model.onnx exists and run from .venv312.")
        sys.exit(1)

    try:
        from training.chromatic_scorer import ChromaticScorer
    except Exception as exc:
        print(f"ERROR: Failed to import ChromaticScorer: {exc}")
        print("Use .venv312/bin/python — ChromaticScorer requires torch + numpy 1.x.")
        sys.exit(1)

    scorer = ChromaticScorer(str(onnx_path))
    concept_emb = scorer.prepare_concept(plan_concept) if plan_concept else None
    target = get_chromatic_target(plan_color)

    edited_text = lyrics_path.read_text(encoding="utf-8")
    edited_results = scorer.score_batch(
        [{"lyric_text": edited_text}], concept_emb=concept_emb
    )
    edited_match = compute_chromatic_match(edited_results[0], target)

    result = {"lyrics_edited_chromatic_match": round(float(edited_match), 4)}

    if not draft_path.exists():
        print(
            "WARNING: melody/lyrics_draft.txt not found — draft score omitted.\n"
            "  (draft is written automatically at promotion time)"
        )
        return result

    draft_text = draft_path.read_text(encoding="utf-8")
    draft_results = scorer.score_batch(
        [{"lyric_text": draft_text}], concept_emb=concept_emb
    )
    draft_match = compute_chromatic_match(draft_results[0], target)

    result["lyrics_draft_chromatic_match"] = round(float(draft_match), 4)
    result["lyrics_chromatic_delta"] = round(
        float(edited_match) - float(draft_match), 4
    )

    return result


# ---------------------------------------------------------------------------
# Rescore (optional --rescore flag)
# ---------------------------------------------------------------------------


def _rescore_phases(
    report: EvaluationReport, production_dir: Path, plan_concept: str
) -> None:
    """Re-score promoted MIDIs using ChromaticScorer. Mutates report in-place."""
    import importlib.util

    scorer_path = (
        Path(__file__).parent.parent.parent.parent / "training" / "chromatic_scorer.py"
    )
    onnx_path = (
        Path(__file__).parent.parent.parent.parent
        / "training"
        / "data"
        / "fusion_model.onnx"
    )
    if not scorer_path.exists() or not onnx_path.exists():
        print("WARNING: ChromaticScorer or ONNX model not found; skipping rescore")
        return

    spec = importlib.util.spec_from_file_location("chromatic_scorer", scorer_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ChromaticScorer = mod.ChromaticScorer

    scorer = ChromaticScorer(str(onnx_path))
    concept_emb = scorer.prepare_concept(plan_concept) if plan_concept else None

    for phase, pr in report.phases.items():
        if not pr.complete:
            continue
        approved_dir = production_dir / phase / "approved"
        if not approved_dir.exists():
            continue
        midi_files = sorted(approved_dir.glob("*.mid"))
        if not midi_files:
            continue
        results = scorer.score_batch(
            [f.read_bytes() for f in midi_files],
            concept_emb=concept_emb,
        )
        match_scores = [r.get("match", 0.0) for r in results]
        confidence_scores = [r.get("confidence", 0.0) for r in results]
        pr.mean_chromatic_match = _safe_mean(match_scores)
        pr.mean_chromatic_confidence = _safe_mean(confidence_scores)
        pr.chromatic_consistency = _chromatic_consistency(match_scores)


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------


def compare_songs(production_dirs: list) -> list:
    """Evaluate all songs and return reports sorted by composite score (desc)."""
    reports = []
    for d in production_dirs:
        try:
            r = evaluate(Path(d))
            reports.append(r)
        except Exception as exc:
            print(f"WARNING: Failed to evaluate {d}: {exc}")
    reports.sort(key=lambda r: r.composite_score, reverse=True)
    return reports


def _print_comparison_table(reports: list) -> None:
    slug_width = max((len(r.song_slug) for r in reports), default=20)
    slug_width = max(slug_width, 20)
    header = f"{'#':>3}  {'Song':<{slug_width}}  {'Score':>7}  {'Readiness':<10}  Flags"
    print()
    print("SONG COMPARISON (ranked by composite score)")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for i, r in enumerate(reports, 1):
        top_flags = ", ".join(r.flags[:2]) if r.flags else "-"
        print(
            f"{i:>3}  {r.song_slug:<{slug_width}}  {r.composite_score:>7.4f}"
            f"  {r.airgigs_readiness:<10}  {top_flags}"
        )
    print()


def _write_comparison_report(reports: list) -> Path:
    """Write comparison_report.yml to cwd."""
    data = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "songs": [
            {
                "rank": i + 1,
                "song_slug": r.song_slug,
                "color": r.color,
                "composite_score": round(r.composite_score, 4),
                "airgigs_readiness": r.airgigs_readiness,
                "phases_complete": r.phases_complete,
                "chromatic_alignment": round(r.chromatic_alignment, 4),
                "theory_quality": round(r.theory_quality, 4),
                "flags": r.flags,
            }
            for i, r in enumerate(reports)
        ],
    }
    out_path = Path(COMPARISON_FILENAME)
    with open(out_path, "w") as f:
        yaml.dump(
            data, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Song Evaluator — AirGigs Readiness Metric"
    )
    parser.add_argument(
        "production_dirs",
        nargs="*",
        help="Production directory/directories to evaluate",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Rank multiple songs and write comparison_report.yml",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Re-run ChromaticScorer on promoted MIDIs (requires ONNX model)",
    )
    parser.add_argument(
        "--rescore-lyrics",
        action="store_true",
        help="Score melody/lyrics.txt and lyrics_draft.txt, merge results into song_evaluation.yml",
    )
    args = parser.parse_args()

    # Expand glob patterns
    dirs: list = []
    for pattern in args.production_dirs:
        expanded = globmod.glob(pattern)
        if expanded:
            dirs.extend(expanded)
        else:
            dirs.append(pattern)

    if not dirs:
        parser.print_help()
        sys.exit(1)

    if args.compare:
        # Evaluate all and print ranked table
        reports = []
        for d in dirs:
            prod_path = Path(d)
            if not prod_path.exists():
                print(f"WARNING: Not found: {prod_path}")
                continue
            try:
                r = evaluate(prod_path)
                if args.rescore:
                    plan = load_plan(prod_path)
                    concept = plan.concept if plan else ""
                    _rescore_phases(r, prod_path, concept)
                    # Recompute derived scores after rescore
                    completed = [pr for pr in r.phases.values() if pr.complete]
                    r.chromatic_alignment = _safe_mean(
                        [pr.mean_chromatic_match for pr in completed]
                    )
                    r.composite_score = _compute_composite(
                        r.chromatic_alignment,
                        r.theory_quality,
                        r.production_completeness,
                        r.structural_integrity,
                        r.lyric_maturity,
                    )
                    r.airgigs_readiness = _determine_readiness(r.composite_score)
                    r.flags = _collect_flags(r)
                reports.append(r)
            except Exception as exc:
                print(f"WARNING: Failed to evaluate {d}: {exc}")
        reports.sort(key=lambda r: r.composite_score, reverse=True)
        _print_comparison_table(reports)
        out = _write_comparison_report(reports)
        print(f"Comparison report written: {out}")
        return

    # Single-song mode
    if len(dirs) != 1:
        print("ERROR: Provide exactly one production_dir for single-song mode")
        print("Use --compare for multiple songs")
        sys.exit(1)

    prod_path = Path(dirs[0])
    if not prod_path.exists():
        print(f"ERROR: Production directory not found: {prod_path}")
        sys.exit(1)

    report = evaluate(prod_path)

    if args.rescore:
        plan = load_plan(prod_path)
        concept = plan.concept if plan else ""
        _rescore_phases(report, prod_path, concept)
        completed = [pr for pr in report.phases.values() if pr.complete]
        report.chromatic_alignment = _safe_mean(
            [pr.mean_chromatic_match for pr in completed]
        )
        report.composite_score = _compute_composite(
            report.chromatic_alignment,
            report.theory_quality,
            report.production_completeness,
            report.structural_integrity,
            report.lyric_maturity,
        )
        report.airgigs_readiness = _determine_readiness(report.composite_score)
        report.flags = _collect_flags(report)
        _write_evaluation(report, prod_path)

    if args.rescore_lyrics:
        plan = load_plan(prod_path)
        concept = plan.concept if plan else ""
        color = plan.color if plan else ""
        lyric_scores = _rescore_lyrics(prod_path, concept, color)
        if lyric_scores:
            eval_path = prod_path / EVALUATION_FILENAME
            existing: dict = {}
            if eval_path.exists():
                with open(eval_path) as f:
                    existing = yaml.safe_load(f) or {}
            existing.update(lyric_scores)
            with open(eval_path, "w") as f:
                yaml.dump(
                    existing,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
            print("\nLyric rescore results:")
            for k, v in lyric_scores.items():
                print(f"  {k}: {v}")
            print(f"\nUpdated: {eval_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"SONG EVALUATION: {report.title or report.song_slug}")
    print(f"{'='*60}")
    print(f"Color:          {report.color}")
    print(f"Phases:         {report.phases_complete}/4 complete")
    print(f"Total bars:     {report.total_bars}")
    print(f"Vocal coverage: {report.vocal_coverage:.1%}")
    print(f"Has lyrics:     {report.has_lyrics}")
    print("")
    print(f"Chromatic alignment:    {report.chromatic_alignment:.4f}")
    print(f"Theory quality:         {report.theory_quality:.4f}")
    print(f"Production completeness:{report.production_completeness:.4f}")
    print(f"Structural integrity:   {report.structural_integrity:.4f}")
    print(f"Lyric maturity:         {report.lyric_maturity:.4f}")
    print("")
    print(f"COMPOSITE SCORE:  {report.composite_score:.4f}")
    print(f"AIRGIGS READINESS: {report.airgigs_readiness.upper()}")
    if report.flags:
        print("\nFlags:")
        for flag in report.flags:
            print(f"  • {flag}")
    out_path = prod_path / EVALUATION_FILENAME
    print(f"\nEvaluation written: {out_path}")


if __name__ == "__main__":
    main()
