#!/usr/bin/env python3
"""
Lyric negative constraints — word/imagery frequency analysis across an album.

Scans promoted `melody/lyrics.txt` files across an album's song productions and
flags words that recur across an unhealthy fraction of songs (e.g. "blue", "dead").
This mirrors the frequency/threshold/severity pattern in
`white_extraction.util.generate_negative_constraints`, but operates on lyric text
rather than song-proposal metadata (key, BPM, title, concept), and is consumed by
the lyric pipeline rather than the White ideation agent. The two mechanisms are
intentionally independent — this module never reads or writes
`negative_constraints.yml`.

Usage:
    python -m white_generation.lyric_negative_constraints \
        --album-dir shrink_wrapped

    python -m white_generation.lyric_negative_constraints --dry-run
"""

import argparse
import os
import re
from collections import Counter
from pathlib import Path

import yaml

# Threshold: flag a word if it appears in more than this fraction of scanned songs
OVERUSE_THRESHOLD = 0.30

# Only flag short/monosyllabic content words — this is the failure mode reported
# (candidates converging on blunt monosyllables like "blue", "dead", "gone").
SHORT_WORD_MAX_SYLLABLES = 1

# Common function words excluded from frequency analysis so the tool flags
# concrete/content words, not grammatical scaffolding.
_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "so",
    "as",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "is",
    "was",
    "are",
    "were",
    "be",
    "been",
    "am",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "its",
    "our",
    "their",
    "this",
    "that",
    "not",
    "no",
    "yes",
    "do",
    "did",
    "does",
    "for",
    "with",
    "from",
    "up",
    "down",
    "out",
    "off",
    "into",
    "over",
    "than",
    "then",
    "now",
    "here",
    "there",
    "all",
    "just",
    "can",
    "will",
    "would",
    "could",
    "should",
}

_HEADER_RE = re.compile(r"^\[[^\]]+\]\s*$")


def _count_syllables(word: str) -> int:
    """Vowel-cluster syllable heuristic — matches lyric_pipeline._count_syllables."""
    return max(1, len(re.findall(r"[aeiouAEIOU]+", word)))


def _tokenize(text: str) -> list[str]:
    """Strip comment/header lines and split remaining text into lowercase words."""
    words = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or _HEADER_RE.match(stripped):
            continue
        words.extend(re.findall(r"[a-zA-Z']+", stripped.lower()))
    return words


def collect_lyric_texts(album_dir: Path) -> list[dict]:
    """Walk an album directory for promoted `melody/lyrics.txt` files.

    Returns a list of {"song_id", "path", "text"} dicts, one per song with a
    promoted lyrics file. Songs without promoted lyrics are skipped silently.
    """
    results = []
    for lyrics_path in sorted(album_dir.glob("*/production/*/melody/lyrics.txt")):
        parts = lyrics_path.parts
        thread_slug = parts[-5]
        production_slug = parts[-3]
        text = lyrics_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        results.append(
            {
                "song_id": f"{thread_slug}__{production_slug}",
                "path": str(lyrics_path),
                "text": text,
            }
        )
    return results


def analyze_word_frequency(
    song_texts: list[dict], threshold: float = OVERUSE_THRESHOLD
) -> dict:
    """Compute per-word document frequency (fraction of songs containing the word).

    Only short/monosyllabic, non-stopword content words are considered — these are
    the words that converge across candidates when phrase syllable targets are tight.
    """
    total = len(song_texts)
    if total == 0:
        return {"overused_words": [], "word_song_counts": {}}

    song_word_counts: Counter = Counter()
    for entry in song_texts:
        words_in_song = set(_tokenize(entry["text"]))
        for word in words_in_song:
            if word in _STOPWORDS or len(word) < 3:
                continue
            if _count_syllables(word) > SHORT_WORD_MAX_SYLLABLES:
                continue
            song_word_counts[word] += 1

    overused = []
    for word, count in song_word_counts.most_common():
        fraction = count / total
        if fraction >= threshold:
            overused.append(
                {
                    "word": word,
                    "count": count,
                    "fraction": round(fraction, 2),
                    "severity": "avoid",
                    "reason": f"'{word}' appears in {count}/{total} songs' lyrics ({fraction:.0%})",
                }
            )

    return {
        "overused_words": overused,
        "word_song_counts": dict(song_word_counts.most_common()),
    }


def generate_constraints(album_dir: Path, threshold: float = OVERUSE_THRESHOLD) -> dict:
    """Generate lyric negative constraints for an album.

    Returns a dict ready to write as YAML, containing `overused_words`, a
    `generated_from` path, and `song_count`. Albums with fewer than 2 songs with
    promoted lyrics still get a valid (empty) constraints file plus a note.
    """
    song_texts = collect_lyric_texts(album_dir)
    analysis = analyze_word_frequency(song_texts, threshold=threshold)

    constraints: dict = {
        "generated_from": str(album_dir),
        "song_count": len(song_texts),
        "overused_words": analysis["overused_words"],
        "word_song_counts": analysis["word_song_counts"],
    }

    if len(song_texts) < 2:
        constraints["note"] = (
            f"Only {len(song_texts)} song(s) with promoted lyrics found — "
            "too few for meaningful frequency analysis"
        )

    return constraints


def format_for_prompt(constraints: dict) -> str:
    """Format constraints as a text block suitable for injection into a lyric prompt."""
    overused = constraints.get("overused_words", [])
    if not overused:
        return ""

    lines = ["## WORDS TO AVOID (overused across the album)", ""]
    for entry in overused:
        lines.append(f"- \"{entry['word']}\" — {entry['reason']}")
    lines.append("")
    lines.append(
        "These words have converged across prior songs' lyrics. Reach for fresh "
        "language and imagery instead of defaulting to this list."
    )
    return "\n".join(lines)


def write_constraints(output_path: Path, constraints: dict) -> Path:
    with open(output_path, "w") as f:
        yaml.dump(
            constraints,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=float("inf"),
        )
    return output_path


def load_constraints(album_dir: Path) -> dict | None:
    """Load lyrics_negative_constraints.yml from an album root, or None if absent."""
    path = album_dir / "lyrics_negative_constraints.yml"
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f) or None


def main():
    parser = argparse.ArgumentParser(
        description="Generate lyric-scoped negative constraints from promoted lyrics"
    )
    _sw_dir = os.getenv("SHRINKWRAP_OUTPUT_DIR", "shrink_wrapped")
    parser.add_argument(
        "--album-dir",
        type=Path,
        default=Path(_sw_dir),
        help="Album (shrink_wrapped) root directory (default: $SHRINKWRAP_OUTPUT_DIR)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <album-dir>/lyrics_negative_constraints.yml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print constraints without writing"
    )

    args = parser.parse_args()
    output_path = args.output or (args.album_dir / "lyrics_negative_constraints.yml")

    constraints = generate_constraints(args.album_dir)

    print(f"Scanned {constraints['song_count']} song(s) with promoted lyrics")
    print(f"Overused words: {len(constraints['overused_words'])}")
    for entry in constraints["overused_words"]:
        print(f"  - {entry['reason']}")
    if constraints.get("note"):
        print(f"Note: {constraints['note']}")

    if args.dry_run:
        print("\n--- Prompt block preview ---")
        preview = format_for_prompt(constraints)
        print(preview or "(no overused words — nothing to inject)")
        return

    path = write_constraints(output_path, constraints)
    print(f"\nWrote constraints to {path}")


if __name__ == "__main__":
    main()
