"""Production plan drift report.

Compares Claude's proposed arrangement (production_plan.yml) against the human's
actual Logic arrangement (arrangement.txt) and writes plan_drift_report.yml with
section drift, bar count deltas, energy arc correlation, and a prose summary.

Usage:
    python -m white_composition.drift_report --production-dir <dir>
    python -m white_composition.drift_report --production-dir <dir> --no-claude
    python -m white_composition.drift_report --production-dir <dir> --arrangement custom.txt
"""

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel

from white_composition.production_plan import (
    PLAN_FILENAME,
    _infer_arc_from_label,
    _parse_time_sig,
    load_plan,
    parse_arrangement_sections,
)

REPORT_FILENAME = "plan_drift_report.yml"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class BarDelta(BaseModel):
    proposed: int
    actual: int
    delta: int


class DriftSummary(BaseModel):
    removed: list[str]
    added: list[str]
    reordered: bool


class DriftReport(BaseModel):
    generated: str
    song_title: str
    proposed_sections: list[str]
    actual_sections: list[str]
    drift: DriftSummary
    bar_deltas: dict[str, BarDelta]
    energy_arc_correlation: Optional[float]
    summary: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_label(name: str) -> str:
    """Normalise a clip or section name to a canonical label.

    Strips trailing version/numeric suffixes (e.g. _v2, _02, _2) so that
    Logic clip names like 'chorus_02' match plan labels like 'chorus'.
    """
    s = name.strip().lower().replace("-", "_")
    s = re.sub(r"_v?\d+$", "", s)
    return s


def _pearson_r(xs: list[float], ys: list[float]) -> Optional[float]:
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    den_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if den_x == 0 or den_y == 0:
        return None
    return round(num / (den_x * den_y), 4)


def _interpolate_to_n(values: list[float], n: int) -> list[float]:
    if len(values) == n:
        return list(values)
    result = []
    for i in range(n):
        pos = i * (len(values) - 1) / (n - 1)
        lo = int(pos)
        hi = min(lo + 1, len(values) - 1)
        frac = pos - lo
        result.append(values[lo] * (1 - frac) + values[hi] * frac)
    return result


def _arc_correlation(
    proposed_arc: list[float], actual_arc: list[float]
) -> Optional[float]:
    """Pearson r between proposed and actual arc trajectories.

    Both sequences are normalised to 100 sample points before comparison so
    that different-length arrangements can be compared meaningfully.
    """
    if len(proposed_arc) < 2 or len(actual_arc) < 2:
        return None
    p = _interpolate_to_n(proposed_arc, 100)
    a = _interpolate_to_n(actual_arc, 100)
    return _pearson_r(p, a)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _expand_proposed(plan) -> list[tuple[str, float, int]]:
    """Expand plan sections by play_count into a flat instance list.

    Returns list of (label, arc, bars) tuples in proposed playback order.
    """
    instances = []
    for sec in plan.sections:
        count = max(1, getattr(sec, "play_count", 1))
        for _ in range(count):
            instances.append((sec.name, float(sec.arc), int(sec.bars)))
    return instances


def _build_drift_summary(
    proposed_instances: list[tuple[str, float, int]],
    actual_instances: list[dict],
) -> DriftSummary:
    proposed_labels = [_normalize_label(n) for n, _, _ in proposed_instances]
    actual_labels = [_normalize_label(i["section_name"]) for i in actual_instances]

    proposed_set = set(proposed_labels)
    actual_set = set(actual_labels)

    removed = sorted(proposed_set - actual_set)
    added = sorted(actual_set - proposed_set)

    # Reordered: compare first-occurrence order of shared labels
    shared = proposed_set & actual_set
    proposed_order = [lb for lb in dict.fromkeys(proposed_labels) if lb in shared]
    actual_order = [lb for lb in dict.fromkeys(actual_labels) if lb in shared]
    reordered = proposed_order != actual_order

    return DriftSummary(removed=removed, added=added, reordered=reordered)


def _build_bar_deltas(
    proposed_instances: list[tuple[str, float, int]],
    actual_instances: list[dict],
) -> dict[str, BarDelta]:
    """Per-label total bar counts for labels present in both proposed and actual."""
    proposed_bars: dict[str, int] = {}
    for name, _, bars in proposed_instances:
        key = _normalize_label(name)
        proposed_bars[key] = proposed_bars.get(key, 0) + bars

    actual_bars: dict[str, int] = {}
    for inst in actual_instances:
        key = _normalize_label(inst["section_name"])
        actual_bars[key] = actual_bars.get(key, 0) + inst["bars"]

    shared = set(proposed_bars) & set(actual_bars)
    return {
        label: BarDelta(
            proposed=proposed_bars[label],
            actual=actual_bars[label],
            delta=actual_bars[label] - proposed_bars[label],
        )
        for label in sorted(shared)
    }


def _build_arc_sequences(
    proposed_instances: list[tuple[str, float, int]],
    actual_instances: list[dict],
) -> tuple[list[float], list[float]]:
    """Build arc value sequences for proposed and actual arrangements."""
    proposed_arc = [arc for _, arc, _ in proposed_instances]

    # Arc lookup from proposed instances (same data, avoids re-reading plan)
    arc_by_label: dict[str, float] = {
        _normalize_label(name): arc for name, arc, _ in proposed_instances
    }
    actual_arc = [
        arc_by_label.get(
            _normalize_label(i["section_name"]),
            _infer_arc_from_label(i["section_name"]),
        )
        for i in actual_instances
    ]
    return proposed_arc, actual_arc


def _generate_summary(report: DriftReport) -> str:
    """Call Claude for a one-paragraph prose interpretation of the drift."""
    try:
        from anthropic import Anthropic
        from dotenv import load_dotenv

        load_dotenv()
        client = Anthropic()

        removed_str = (
            ", ".join(report.drift.removed) if report.drift.removed else "none"
        )
        added_str = ", ".join(report.drift.added) if report.drift.added else "none"
        reordered_str = "yes" if report.drift.reordered else "no"
        arc_str = (
            f"{report.energy_arc_correlation:.2f}"
            if report.energy_arc_correlation is not None
            else "n/a"
        )
        deltas = "\n".join(
            f"  {label}: proposed {d.proposed} bars → actual {d.actual} bars (Δ{d.delta:+d})"
            for label, d in report.bar_deltas.items()
        )

        prompt = f"""You are a production analyst for the White Project — an AI-driven music composition system.

Claude proposed a song arrangement and the human then arranged it in Logic Pro. Here is the drift data:

Song: {report.song_title}
Proposed sections: {', '.join(report.proposed_sections)}
Actual sections: {', '.join(report.actual_sections)}
Removed: {removed_str}
Added: {added_str}
Reordered: {reordered_str}
Energy arc correlation: {arc_str}

Bar count deltas per section:
{deltas if deltas else "  (no shared sections with deltas)"}

Write a single concise paragraph (3–5 sentences) interpreting what the human changed, why those changes might have been made, and what this implies about Claude's compositional judgement for future proposals. Be specific and direct — treat the divergence as a learning signal, not a failure."""

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"  Warning: Claude summary unavailable ({e})")
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_plans(
    plan,
    arrangement_path: Path,
    use_claude: bool = True,
) -> DriftReport:
    """Compare a ProductionPlan against an arrangement.txt export.

    Args:
        plan: ProductionPlan loaded from production_plan.yml
        arrangement_path: Path to arrangement.txt
        use_claude: Whether to call Claude for the prose summary

    Returns:
        DriftReport with all drift fields populated
    """
    num, den = _parse_time_sig(plan.time_sig or "4/4")
    beats_per_bar = num * (4 // den)
    actual_instances = parse_arrangement_sections(
        arrangement_path,
        bpm=float(plan.bpm or 120),
        beats_per_bar=beats_per_bar,
    )

    proposed_instances = _expand_proposed(plan)

    proposed_sections = [name for name, _, _ in proposed_instances]
    actual_sections = [i["section_name"] for i in actual_instances]

    drift = _build_drift_summary(proposed_instances, actual_instances)
    bar_deltas = _build_bar_deltas(proposed_instances, actual_instances)
    proposed_arc, actual_arc = _build_arc_sequences(
        proposed_instances, actual_instances
    )
    arc_corr = _arc_correlation(proposed_arc, actual_arc)

    report = DriftReport(
        generated=datetime.now(timezone.utc).isoformat(),
        song_title=plan.title or plan.song_slug,
        proposed_sections=proposed_sections,
        actual_sections=actual_sections,
        drift=drift,
        bar_deltas=bar_deltas,
        energy_arc_correlation=arc_corr,
        summary="",
    )

    if use_claude:
        try:
            report.summary = _generate_summary(report)
        except Exception as e:
            print(f"  Warning: summary generation failed ({e})")

    return report


def write_report(production_dir: Path, report: DriftReport) -> Path:
    """Write plan_drift_report.yml to the production directory."""
    out_path = production_dir / REPORT_FILENAME
    data = {
        "generated": report.generated,
        "song_title": report.song_title,
        "proposed_sections": report.proposed_sections,
        "actual_sections": report.actual_sections,
        "drift": {
            "removed": report.drift.removed,
            "added": report.drift.added,
            "reordered": report.drift.reordered,
        },
        "bar_deltas": {
            label: {"proposed": d.proposed, "actual": d.actual, "delta": d.delta}
            for label, d in report.bar_deltas.items()
        },
        "energy_arc_correlation": report.energy_arc_correlation,
        "summary": report.summary,
    }
    with open(out_path, "w") as f:
        yaml.dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=float("inf"),
        )
    return out_path


def load_report(production_dir: Path) -> Optional[DriftReport]:
    """Load plan_drift_report.yml from the production directory, or None if absent."""
    path = production_dir / REPORT_FILENAME
    if not path.exists():
        return None
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    bar_deltas = {
        label: BarDelta(**vals)
        for label, vals in (data.get("bar_deltas") or {}).items()
    }
    drift_raw = data.get("drift") or {}
    return DriftReport(
        generated=str(data.get("generated", "")),
        song_title=str(data.get("song_title", "")),
        proposed_sections=list(data.get("proposed_sections") or []),
        actual_sections=list(data.get("actual_sections") or []),
        drift=DriftSummary(
            removed=list(drift_raw.get("removed") or []),
            added=list(drift_raw.get("added") or []),
            reordered=bool(drift_raw.get("reordered", False)),
        ),
        bar_deltas=bar_deltas,
        energy_arc_correlation=data.get("energy_arc_correlation"),
        summary=str(data.get("summary") or ""),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate plan_drift_report.yml comparing production_plan.yml "
        "against arrangement.txt"
    )
    parser.add_argument(
        "--production-dir", required=True, help="Song production directory"
    )
    parser.add_argument(
        "--arrangement",
        default=None,
        help="Path to arrangement.txt (default: <production-dir>/arrangement.txt)",
    )
    parser.add_argument(
        "--no-claude",
        action="store_true",
        help="Skip Claude summary (faster, offline-safe)",
    )
    args = parser.parse_args()

    prod_path = Path(args.production_dir)
    if not prod_path.exists():
        print(f"ERROR: Production directory not found: {prod_path}")
        sys.exit(1)

    plan = load_plan(prod_path)
    if plan is None:
        print(f"ERROR: {PLAN_FILENAME} not found in {prod_path}")
        sys.exit(1)

    arr_path = (
        Path(args.arrangement) if args.arrangement else prod_path / "arrangement.txt"
    )
    if not arr_path.exists():
        print(f"ERROR: arrangement.txt not found at {arr_path}")
        sys.exit(1)

    print("=" * 60)
    print("PLAN DRIFT REPORT")
    print("=" * 60)
    print(f"Song:    {plan.title or plan.song_slug}")

    report = compare_plans(plan, arr_path, use_claude=not args.no_claude)
    out_path = write_report(prod_path, report)

    print(
        f"Proposed: {len(report.proposed_sections)} instances — {report.proposed_sections}"
    )
    print(
        f"Actual:   {len(report.actual_sections)} instances — {report.actual_sections}"
    )
    if report.drift.removed:
        print(f"Removed:  {report.drift.removed}")
    if report.drift.added:
        print(f"Added:    {report.drift.added}")
    if report.drift.reordered:
        print("Reordered: yes")
    if report.bar_deltas:
        print("\nBar deltas:")
        for label, d in report.bar_deltas.items():
            if d.delta:
                print(
                    f"  {label:<20} proposed={d.proposed:>3}  actual={d.actual:>3}  Δ{d.delta:+d}"
                )
    if report.energy_arc_correlation is not None:
        print(f"\nArc correlation: {report.energy_arc_correlation:.2f}")
    if report.summary:
        print(f"\nSummary:\n{report.summary}")
    print(f"\nReport written: {out_path}")


if __name__ == "__main__":
    main()
