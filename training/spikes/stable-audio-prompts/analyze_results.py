#!/usr/bin/env python3
"""
Analyze spike_results.jsonl and write spike_report.md.

Usage:
    python training/spikes/stable-audio-prompts/analyze_results.py \
        --results spike_output/spike_results.jsonl \
        --corpus-baseline  # compute corpus baseline from local parquet
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

PASS_THRESHOLD = 0.4
REPORT_PATH = Path(__file__).parent / "spike_report.md"

CHROMATIC_PROMPTS = {
    "Red": (
        "sparse ambient piano, melancholic, slow, introspective, past memories, "
        "wistful, fading, quiet reverb, nostalgic"
    ),
    "Orange": (
        "warm nostalgic folk texture, familiar place, rootsy acoustic guitar, "
        "gentle, grounded, orange sunset, belonging"
    ),
    "Yellow": (
        "present moment, bright acoustic texture, open fields, optimistic, "
        "forward motion, daylight, organic, alive"
    ),
    "Green": (
        "organic nature sounds, present-tense growth, living breathing texture, "
        "forest ambience, gentle movement, verdant"
    ),
    "Blue": (
        "present urban ambient, cool minimal electronic, observational, "
        "street sounds dissolved into drone, detached, blue hour"
    ),
    "Indigo": (
        "alien drone, suspended time, no tonal anchor, unreal space, "
        "dissonant shimmer, liminal void, indigo darkness"
    ),
    "Violet": (
        "dreaming, liminal between-states, soft electronic wash, future imagined, "
        "violet dusk, floating, aspirational, dissolving"
    ),
    "Black": (
        "dark ambient void, formless, subterranean rumble, undefined, no melody, "
        "deep silence with texture, black space"
    ),
}


def load_results(path: str) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def compute_corpus_baseline(top_n: int = 20) -> dict[str, float]:
    """Compute mean chromatic_match for top-N retrieved segments per color."""
    try:
        from white_composition.retrieve_samples import (
            load_clap_index,
            retrieve_by_color,
        )

        print("Computing corpus baseline from local CLAP parquet...")
        df = load_clap_index()
        baseline = {}
        for color in CHROMATIC_PROMPTS:
            results = retrieve_by_color(df, color, top_n=top_n)
            if results:
                baseline[color] = sum(r["match"] for r in results) / len(results)
            else:
                baseline[color] = 0.0
            print(f"  {color}: {baseline[color]:.4f}")
        return baseline
    except Exception as e:
        print(f"  Could not compute corpus baseline: {e}")
        return {}


def analyze(results: list[dict], baseline: dict[str, float]) -> dict:
    by_color: dict[str, list[float]] = defaultdict(list)
    times: dict[str, list[float]] = defaultdict(list)

    for r in results:
        if r.get("chromatic_match") is not None and not r.get("dry_run"):
            by_color[r["color"]].append(r["chromatic_match"])
            times[r["color"]].append(r.get("generation_time_s", 0))

    summary = {}
    for color in CHROMATIC_PROMPTS:
        scores = by_color.get(color, [])
        if not scores:
            summary[color] = None
            continue
        mean = sum(scores) / len(scores)
        pass_rate = sum(1 for s in scores if s > PASS_THRESHOLD) / len(scores)
        mean_time = sum(times[color]) / len(times[color]) if times[color] else 0
        corp = baseline.get(color)
        summary[color] = {
            "n": len(scores),
            "mean": round(mean, 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "pass_rate": round(pass_rate, 3),
            "vs_corpus": round(mean - corp, 4) if corp else None,
            "mean_time_s": round(mean_time, 1),
            "scores": scores,
        }
    return summary


def go_nogo(summary: dict) -> tuple[str, str]:
    """Determine go/no-go based on overall pass rate across all colors."""
    all_pass_rates = [v["pass_rate"] for v in summary.values() if v]
    if not all_pass_rates:
        return "no-go", "No results to evaluate."

    overall = sum(all_pass_rates) / len(all_pass_rates)
    colors_passing = sum(1 for r in all_pass_rates if r >= 0.5)

    if overall >= 0.5:
        verdict = "go"
        rationale = (
            f"Overall pass rate {overall:.0%} meets the ≥50% threshold. "
            f"{colors_passing}/{len(all_pass_rates)} colors achieve ≥50% pass rate. "
            "Text-conditioned synthesis is a viable complement to granular retrieval."
        )
    else:
        verdict = "no-go"
        rationale = (
            f"Overall pass rate {overall:.0%} is below the ≥50% threshold. "
            f"Only {colors_passing}/{len(all_pass_rates)} colors meet ≥50%. "
            "Text prompts alone are not a reliable chromatic targeting mechanism — "
            "granular retrieval remains the primary path."
        )

    return verdict, rationale


def write_report(summary: dict, baseline: dict, results: list[dict]) -> Path:
    verdict, rationale = go_nogo(summary)

    lines = [
        "# Spike Report: Stable Audio Open — Chromatic Text Prompt Evaluation",
        "",
        f"**Date:** {__import__('datetime').date.today().isoformat()}",
        "**Model:** `stabilityai/stable-audio-open-1.0` (diffusers, StableAudioPipeline)",
        "**GPU:** Modal A10G",
        f"**Clips:** {len([r for r in results if not r.get('dry_run')])} total "
        f"({len([r for r in results if not r.get('dry_run')]) // len(CHROMATIC_PROMPTS)} per color × {len(CHROMATIC_PROMPTS)} colors)",
        f"**Verdict:** **{verdict.upper()}**",
        "",
        "---",
        "",
        "## 1. Prompts Used",
        "",
    ]

    for color, prompt in CHROMATIC_PROMPTS.items():
        lines.append(f"**{color}:** {prompt}")
        lines.append("")

    lines += [
        "---",
        "",
        "## 2. Results",
        "",
        f"Pass threshold: chromatic_match > {PASS_THRESHOLD}",
        "",
        "| Color | Mean match | Min | Max | Pass rate | vs corpus baseline | Gen time |",
        "|-------|-----------|-----|-----|-----------|-------------------|---------|",
    ]

    for color in CHROMATIC_PROMPTS:
        s = summary.get(color)
        if not s:
            lines.append(f"| {color} | — | — | — | — | — | — |")
            continue
        vs = f"{s['vs_corpus']:+.4f}" if s["vs_corpus"] is not None else "—"
        lines.append(
            f"| {color} | {s['mean']:.4f} | {s['min']:.4f} | {s['max']:.4f} | "
            f"{s['pass_rate']:.0%} ({sum(1 for sc in s['scores'] if sc > PASS_THRESHOLD)}/{s['n']}) | "
            f"{vs} | {s['mean_time_s']}s |"
        )

    all_pass_rates = [v["pass_rate"] for v in summary.values() if v]
    overall = sum(all_pass_rates) / len(all_pass_rates) if all_pass_rates else 0
    lines += [
        "",
        f"**Overall pass rate:** {overall:.0%}",
        "",
    ]

    if baseline:
        lines += [
            "### Corpus baseline (mean chromatic_match, top-20 retrieved segments)",
            "",
            "| Color | Corpus baseline |",
            "|-------|----------------|",
        ]
        for color, b in baseline.items():
            lines.append(f"| {color} | {b:.4f} |")
        lines.append("")

    lines += [
        "---",
        "",
        "## 3. License Assessment",
        "",
        "Stable Audio Open uses the **Stability AI Community License**:",
        "",
        "- Non-commercial and small-business use (annual revenue < $1M): **free**",
        "- Commercial use above $1M revenue threshold: requires a paid license from Stability AI",
        "- Generated outputs are owned by the operator; no attribution required",
        "",
        "**For Earthly Frames:** If annual revenue is currently under $1M, generated audio can be",
        "used commercially under the community license at no cost. Verify current revenue threshold",
        "at https://stability.ai/community-license-agreement before shipping any generated audio.",
        "",
        "This is meaningfully better than MusicGen (CC-BY-NC = non-commercial only with no pathway",
        "to commercial use).",
        "",
        "---",
        "",
        "## 4. Go / No-Go",
        "",
        f"**{verdict.upper()}**",
        "",
        rationale,
        "",
    ]

    if verdict == "go":
        lines += [
            "---",
            "",
            "## 5. Follow-On Proposal Sketch: `add-stable-audio-synthesis`",
            "",
            "If this spike returns go, the follow-on feature would:",
            "",
            "1. Add `training/tools/stable_audio_synthesizer.py` — wraps `StableAudioPipeline`,",
            "   accepts a color (or full song proposal) and returns a WAV + metadata",
            "2. Integrate with `retrieve_samples.py` pattern: color → prompt → audio → Refractor score",
            "3. CLI: `--color`, `--duration`, `--seed`, `--production-dir` (reads from song_context.yml)",
            "4. Output alongside `grain_synthesizer.py` — two paths: corpus collage (granular) vs.",
            "   novel synthesis (Stable Audio). Human chooses which texture fits the song.",
            "5. Refractor scoring on generated output to confirm chromatic match before use",
            "",
            "**Key difference from granular synthesis:** Stable Audio generates audio that has never",
            "existed — no corpus samples, no copyright from source recordings. Granular synthesis",
            "recombines existing White corpus audio.",
        ]

    report = "\n".join(lines)
    REPORT_PATH.write_text(report)
    print(f"\nReport written: {REPORT_PATH}")
    return REPORT_PATH


def main():
    parser = argparse.ArgumentParser(description="Analyze Stable Audio spike results.")
    parser.add_argument(
        "--results",
        required=True,
        help="Path to spike_results.jsonl",
    )
    parser.add_argument(
        "--corpus-baseline",
        action="store_true",
        help="Compute corpus baseline from local CLAP parquet (requires local data)",
    )
    args = parser.parse_args()

    results = load_results(args.results)
    print(f"Loaded {len(results)} result(s) from {args.results}")

    baseline = compute_corpus_baseline() if args.corpus_baseline else {}
    summary = analyze(results, baseline)

    # Print quick summary
    print(f"\n{'Color':<10} {'Mean':<8} {'Pass rate'}")
    print("-" * 30)
    for color in CHROMATIC_PROMPTS:
        s = summary.get(color)
        if s:
            print(f"{color:<10} {s['mean']:.4f}   {s['pass_rate']:.0%}")
        else:
            print(f"{color:<10} —")

    verdict, rationale = go_nogo(summary)
    print(f"\nVerdict: {verdict.upper()}")
    print(rationale)

    write_report(summary, baseline, results)


if __name__ == "__main__":
    main()
