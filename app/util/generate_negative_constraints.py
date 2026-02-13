"""Generate negative constraints from prior chain results to prevent convergence.

Reads shrinkwrapped/index.yml (or a custom path) and produces negative_constraints.yml
with key/BPM avoidance rules, concept keyword warnings, title exclusions, and diversity metrics.

Usage:
    python -m app.util.generate_negative_constraints                        # default paths
    python -m app.util.generate_negative_constraints --index shrinkwrapped/index.yml
    python -m app.util.generate_negative_constraints --output constraints.yml
    python -m app.util.generate_negative_constraints --dry-run
"""

import argparse
import logging
import math
import re
import yaml

from collections import Counter
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Threshold: if a key appears in more than this fraction of proposals, flag it
KEY_CLUSTER_THRESHOLD = 0.30

# BPM tolerance for clustering
BPM_TOLERANCE = 5

# Common concept phrases to detect (case-insensitive)
CONCEPT_MARKER_PHRASES = [
    "transmigration",
    "information → time → space",
    "information through time into space",
    "seven chromatic methodologies",
    "consciousness archaeology",
    "chromatic convergence",
    "prism synthesis",
    "consciousness examining itself",
    "i am a mirror",
    "narcissistic",
    "sultan of solipsism",
    "rebracketing",
]

# Title words that indicate convergent thinking when overused
TITLE_VOCABULARY_WORDS = [
    "prism",
    "mirror",
    "taxonomy",
    "frequency",
    "field guide",
    "protocol",
    "architect",
    "consciousness",
    "chromatic",
    "convergence",
    "solipsism",
    "sultan",
    "recursive",
]

# Threshold: flag a title word if it appears in more than this fraction of titles
TITLE_WORD_THRESHOLD = 0.20

# Dialogue opener patterns (case-insensitive, checked at start of responses)
DIALOGUE_OPENER_PHRASES = [
    "look,",
    "oh shit",
    "i mean,",
    "honestly,",
    "here's the thing",
    "so here's",
    "okay so",
]

# Threshold: flag a dialogue opener if it appears in more than this fraction of responses
DIALOGUE_OPENER_THRESHOLD = 0.30


def normalize_key(raw_key: str) -> str:
    """Normalize key strings to standard form.

    Handles cases like:
        "C major (resolving through complete chromatic cycle)" -> "C major"
        "C hromatic Complete" -> "C major"  (known typo/variant)
        "A ll Keys (Chromatic Convergence)" -> "All Keys"
        "F# minor" -> "F# minor"
    """
    if not raw_key:
        return "unknown"

    text = raw_key.strip()

    # Known mangled keys from the LLM
    if text.startswith("C hromatic"):
        return "C major"
    if text.startswith("A ll Keys"):
        return "All Keys"

    # Strip parenthetical qualifiers
    text = re.sub(r"\s*\(.*\)$", "", text)
    return text.strip()


def load_index(index_path: Path) -> list[dict]:
    """Load thread metadata from index.yml."""
    if not index_path.exists():
        logger.error(f"Index not found: {index_path}")
        return []

    with open(index_path) as f:
        data = yaml.safe_load(f)

    return data.get("threads", [])


def analyze_keys(threads: list[dict]) -> dict:
    """Analyze key distribution and detect clusters."""
    total = len(threads)
    if total == 0:
        return {"distribution": {}, "clusters": [], "entropy": 0.0}

    keys = [normalize_key(t.get("key", "unknown")) for t in threads]
    counts = Counter(keys)

    # Calculate entropy (bits)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    # Detect clusters above threshold
    clusters = []
    for key, count in counts.most_common():
        fraction = count / total
        if fraction >= KEY_CLUSTER_THRESHOLD:
            severity = "exclude" if fraction >= 0.5 else "avoid"
            clusters.append(
                {
                    "key": key,
                    "count": count,
                    "fraction": round(fraction, 2),
                    "severity": severity,
                    "reason": f"{count}/{total} proposals ({fraction:.0%}) use {key}",
                }
            )

    return {
        "distribution": dict(counts.most_common()),
        "clusters": clusters,
        "entropy": round(entropy, 2),
    }


def analyze_bpm(threads: list[dict]) -> dict:
    """Analyze BPM distribution and detect clusters."""
    bpms = [t.get("bpm") for t in threads if t.get("bpm")]
    if not bpms:
        return {"distribution": {}, "clusters": [], "std_dev": 0.0, "mean": 0.0}

    mean_bpm = sum(bpms) / len(bpms)
    variance = sum((b - mean_bpm) ** 2 for b in bpms) / len(bpms)
    std_dev = math.sqrt(variance)

    # Group BPMs into clusters (within ±BPM_TOLERANCE)
    bpm_counts = Counter(bpms)
    clusters = []
    seen_bpms = set()

    for bpm in sorted(bpm_counts.keys()):
        if bpm in seen_bpms:
            continue

        # Find all BPMs within tolerance
        cluster_bpms = [b for b in bpms if abs(b - bpm) <= BPM_TOLERANCE]
        cluster_count = len(cluster_bpms)
        fraction = cluster_count / len(bpms)

        if fraction >= KEY_CLUSTER_THRESHOLD:
            cluster_center = round(sum(cluster_bpms) / cluster_count)
            severity = "avoid"
            clusters.append(
                {
                    "bpm_range": f"{bpm - BPM_TOLERANCE}-{bpm + BPM_TOLERANCE}",
                    "center": cluster_center,
                    "count": cluster_count,
                    "fraction": round(fraction, 2),
                    "severity": severity,
                    "reason": f"{cluster_count}/{len(bpms)} proposals cluster around {cluster_center} BPM",
                }
            )
            for b in cluster_bpms:
                seen_bpms.add(b)

    return {
        "distribution": dict(bpm_counts.most_common()),
        "clusters": clusters,
        "std_dev": round(std_dev, 2),
        "mean": round(mean_bpm, 1),
    }


def analyze_concepts(threads: list[dict]) -> dict:
    """Analyze concept text for repeated phrases."""
    concepts = [t.get("concept", "") for t in threads]
    total = len(concepts)
    if total == 0:
        return {"repeated_phrases": [], "phrase_counts": {}}

    # Count occurrences of marker phrases
    phrase_counts = {}
    for phrase in CONCEPT_MARKER_PHRASES:
        count = sum(1 for c in concepts if phrase.lower() in c.lower())
        if count > 0:
            phrase_counts[phrase] = count

    # Flag phrases appearing in >30% of concepts
    repeated = []
    for phrase, count in sorted(phrase_counts.items(), key=lambda x: -x[1]):
        fraction = count / total
        if fraction >= KEY_CLUSTER_THRESHOLD:
            repeated.append(
                {
                    "phrase": phrase,
                    "count": count,
                    "fraction": round(fraction, 2),
                    "severity": "avoid",
                    "reason": f"'{phrase}' appears in {count}/{total} concepts ({fraction:.0%})",
                }
            )

    return {
        "repeated_phrases": repeated,
        "phrase_counts": phrase_counts,
    }


def analyze_title_vocabulary(threads: list[dict]) -> dict:
    """Detect overused words/phrases in titles."""
    titles = [t.get("title", "") for t in threads if t.get("title")]
    total = len(titles)
    if total == 0:
        return {"overused_words": [], "word_counts": {}}

    word_counts = {}
    for word in TITLE_VOCABULARY_WORDS:
        count = sum(1 for t in titles if word.lower() in t.lower())
        if count > 0:
            word_counts[word] = count

    overused = []
    for word, count in sorted(word_counts.items(), key=lambda x: -x[1]):
        fraction = count / total
        if fraction >= TITLE_WORD_THRESHOLD:
            overused.append(
                {
                    "word": word,
                    "count": count,
                    "fraction": round(fraction, 2),
                    "severity": "avoid",
                    "reason": f"'{word}' appears in {count}/{total} titles ({fraction:.0%})",
                }
            )

    return {"overused_words": overused, "word_counts": word_counts}


def analyze_dialogue_openers(
    threads: list[dict], shrinkwrap_dir: Optional[Path] = None
) -> dict:
    """Detect repeated dialogue opener patterns in interview/voice outputs.

    Scans markdown files in shrinkwrapped directories for Walsh dialogue responses.
    Falls back to checking thread metadata fields if no markdown files are available.
    """
    # Collect all text that might contain dialogue
    all_responses = []

    # Primary source: scan markdown files in shrinkwrapped directories
    if shrinkwrap_dir and shrinkwrap_dir.exists():
        for t in threads:
            dir_name = t.get("directory", "")
            md_dir = shrinkwrap_dir / dir_name / "md"
            if not md_dir.is_dir():
                continue
            for md_file in md_dir.iterdir():
                if not md_file.suffix == ".md":
                    continue
                try:
                    text = md_file.read_text()
                    parts = re.split(r"\*\*Walsh:\*\*\s*", text)
                    all_responses.extend(parts[1:] if len(parts) > 1 else [])
                except Exception:
                    continue

    # Fallback: check thread metadata fields
    if not all_responses:
        for t in threads:
            for field in ("voice_text", "interview_text", "concept"):
                text = t.get(field, "")
                if text:
                    parts = re.split(r"\*\*Walsh:\*\*\s*", text)
                    all_responses.extend(parts[1:] if len(parts) > 1 else [])

    if not all_responses:
        return {"overused_openers": [], "opener_counts": {}}

    total = len(all_responses)
    opener_counts = {}
    for phrase in DIALOGUE_OPENER_PHRASES:
        count = sum(
            1 for r in all_responses if r.strip().lower().startswith(phrase.lower())
        )
        if count > 0:
            opener_counts[phrase] = count

    overused = []
    for phrase, count in sorted(opener_counts.items(), key=lambda x: -x[1]):
        fraction = count / total
        if fraction >= DIALOGUE_OPENER_THRESHOLD:
            overused.append(
                {
                    "phrase": phrase,
                    "count": count,
                    "fraction": round(fraction, 2),
                    "severity": "avoid",
                    "reason": f"'{phrase}' opens {count}/{total} responses ({fraction:.0%})",
                }
            )

    return {"overused_openers": overused, "opener_counts": opener_counts}


def collect_titles(threads: list[dict]) -> list[str]:
    """Collect all existing titles for exclusion."""
    return sorted(set(t.get("title", "") for t in threads if t.get("title")))


def generate_constraints(
    index_path: Path,
    manual_constraints: Optional[dict] = None,
) -> dict:
    """Generate negative constraints from index data.

    Args:
        index_path: Path to shrinkwrapped/index.yml.
        manual_constraints: Optional dict of manually added constraints to preserve.

    Returns:
        Complete constraints dict ready to write as YAML.
    """
    threads = load_index(index_path)
    if not threads:
        return {"error": "No threads found in index"}

    key_analysis = analyze_keys(threads)
    bpm_analysis = analyze_bpm(threads)
    concept_analysis = analyze_concepts(threads)
    title_vocab_analysis = analyze_title_vocabulary(threads)
    shrinkwrap_dir = index_path.parent if index_path else None
    dialogue_analysis = analyze_dialogue_openers(threads, shrinkwrap_dir)
    titles = collect_titles(threads)

    # Build constraints
    constraints = {
        "generated_from": str(index_path),
        "thread_count": len(threads),
        "key_constraints": key_analysis["clusters"],
        "bpm_constraints": bpm_analysis["clusters"],
        "concept_constraints": concept_analysis["repeated_phrases"],
        "title_vocabulary_constraints": title_vocab_analysis["overused_words"],
        "dialogue_opener_constraints": dialogue_analysis["overused_openers"],
        "excluded_titles": titles,
        "diversity_metrics": {
            "key_entropy": key_analysis["entropy"],
            "key_distribution": key_analysis["distribution"],
            "bpm_std_dev": bpm_analysis["std_dev"],
            "bpm_mean": bpm_analysis["mean"],
            "bpm_distribution": bpm_analysis["distribution"],
            "concept_phrase_counts": concept_analysis["phrase_counts"],
            "title_word_counts": title_vocab_analysis["word_counts"],
            "dialogue_opener_counts": dialogue_analysis["opener_counts"],
        },
        "warnings": [],
    }

    # Generate warnings
    if key_analysis["entropy"] < 2.0:
        constraints["warnings"].append(
            f"Key entropy is {key_analysis['entropy']} bits (threshold: 2.0) — "
            f"proposals are heavily concentrated in a few keys"
        )
    if bpm_analysis["std_dev"] < 10:
        constraints["warnings"].append(
            f"BPM std dev is {bpm_analysis['std_dev']} (threshold: 10) — "
            f"proposals cluster around {bpm_analysis['mean']} BPM"
        )

    # Preserve manual constraints
    if manual_constraints:
        manual_keys = manual_constraints.get("manual_overrides", {})
        if manual_keys:
            constraints["manual_overrides"] = manual_keys

    return constraints


def format_for_prompt(constraints: dict) -> str:
    """Format constraints as a text block suitable for injection into an LLM prompt.

    Returns a concise, readable summary the White Agent can act on.
    """
    lines = ["## NEGATIVE CONSTRAINTS (from prior results)", ""]

    # Key avoidance
    key_constraints = constraints.get("key_constraints", [])
    if key_constraints:
        lines.append("### Keys to AVOID:")
        for kc in key_constraints:
            severity = kc["severity"].upper()
            lines.append(f"- **{severity}** {kc['key']} — {kc['reason']}")
        lines.append("")

    # BPM avoidance
    bpm_constraints = constraints.get("bpm_constraints", [])
    if bpm_constraints:
        lines.append("### BPM ranges to AVOID:")
        for bc in bpm_constraints:
            lines.append(
                f"- **{bc['severity'].upper()}** BPM {bc['bpm_range']} — {bc['reason']}"
            )
        lines.append("")

    # Concept phrases
    concept_constraints = constraints.get("concept_constraints", [])
    if concept_constraints:
        lines.append("### Concept phrases that are OVERUSED (find fresh language):")
        for cc in concept_constraints:
            lines.append(f"- \"{cc['phrase']}\" — {cc['reason']}")
        lines.append("")

    # Title vocabulary
    title_vocab = constraints.get("title_vocabulary_constraints", [])
    if title_vocab:
        lines.append(
            "### Title words that are OVERUSED (do NOT use these in new titles):"
        )
        for tv in title_vocab:
            lines.append(f"- \"{tv['word']}\" — {tv['reason']}")
        lines.append("")

    # Dialogue openers
    dialogue_openers = constraints.get("dialogue_opener_constraints", [])
    if dialogue_openers:
        lines.append("### Dialogue openers that are OVERUSED (vary how Walsh speaks):")
        for do in dialogue_openers:
            lines.append(
                f"- Do NOT start responses with \"{do['phrase']}\" — {do['reason']}"
            )
        lines.append("")

    # Titles
    excluded = constraints.get("excluded_titles", [])
    if excluded:
        lines.append(f"### Titles already used ({len(excluded)} — do NOT reuse):")
        for t in excluded:
            lines.append(f"- {t}")
        lines.append("")

    # Warnings
    warnings = constraints.get("warnings", [])
    if warnings:
        lines.append("### DIVERSITY WARNINGS:")
        for w in warnings:
            lines.append(f"- ⚠ {w}")
        lines.append("")

    lines.append(
        "Use these constraints as creative challenges: "
        "find UNEXPLORED keys, tempos, conceptual territory, and vocal personality."
    )

    return "\n".join(lines)


def write_constraints(output_path: Path, constraints: dict) -> Path:
    """Write constraints to YAML file, preserving manual overrides."""
    # If file exists, load manual overrides
    if output_path.exists():
        with open(output_path) as f:
            existing = yaml.safe_load(f) or {}
        manual = existing.get("manual_overrides")
        if manual:
            constraints["manual_overrides"] = manual

    with open(output_path, "w") as f:
        yaml.dump(
            constraints, f, default_flow_style=False, allow_unicode=True, width=120
        )

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate negative constraints from prior chain results"
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("shrinkwrapped/index.yml"),
        help="Path to shrinkwrapped index.yml (default: shrinkwrapped/index.yml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("shrinkwrapped/negative_constraints.yml"),
        help="Output path for constraints file (default: shrinkwrapped/negative_constraints.yml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print constraints without writing"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    constraints = generate_constraints(args.index)

    if "error" in constraints:
        print(f"Error: {constraints['error']}")
        return

    # Print summary
    metrics = constraints["diversity_metrics"]
    print(f"Analyzed {constraints['thread_count']} threads")
    print(f"  Key entropy: {metrics['key_entropy']} bits")
    print(f"  BPM std dev: {metrics['bpm_std_dev']}")
    print(f"  Key clusters: {len(constraints['key_constraints'])}")
    print(f"  BPM clusters: {len(constraints['bpm_constraints'])}")
    print(f"  Overused phrases: {len(constraints['concept_constraints'])}")
    print(f"  Excluded titles: {len(constraints['excluded_titles'])}")

    if constraints["warnings"]:
        print("\nWarnings:")
        for w in constraints["warnings"]:
            print(f"  ⚠ {w}")

    if args.dry_run:
        print("\n--- Prompt block preview ---")
        print(format_for_prompt(constraints))
        return

    path = write_constraints(args.output, constraints)
    print(f"\nWrote constraints to {path}")


if __name__ == "__main__":
    main()
