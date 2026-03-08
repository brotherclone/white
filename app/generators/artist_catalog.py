#!/usr/bin/env python3
"""
Artist Style Catalog — CLI and shared utilities

Generates and manages aesthetic descriptions for artists referenced in
sounds_like fields, suitable for injection into lyric and chord generation
prompts. Descriptions are Claude-generated (aesthetic only, no biography,
no copyrighted content), then human-reviewed.

Usage:
    # Generate descriptions for all uncatalogued artists in a thread
    python -m app.generators.artist_catalog \
        --thread shrink_wrapped/white-the-breathing-machine-learns-to-sing \
        --generate-missing

    # Also scan training data parquet
    python -m app.generators.artist_catalog \
        --from-training-data --generate-missing

    # Score all reviewed descriptions through Refractor
    python -m app.generators.artist_catalog --score-chromatic

    # Show catalog status
    python -m app.generators.artist_catalog --summary
"""

import argparse
import re
import sys
import yaml

from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

CATALOG_DEFAULT_PATH = (
    Path(__file__).parent.parent / "app" / "reference" / "music" / "artist_catalog.yml"
)
TRAINING_PARQUET_PATH = (
    Path(__file__).parent.parent.parent
    / "training"
    / "data"
    / "training_data_with_embeddings.parquet"
)

_GENERATE_PROMPT_TEMPLATE = """\
Describe the artistic style of {artist_name} in 100–150 words for use as a \
compositional reference. Focus exclusively on:
- Sonic texture and production character (timbre, density, studio approach)
- Lyrical and thematic tendencies (themes, imagery, tone — do NOT reproduce lyrics)
- Emotional register (mood, affect, psychological colour)
- Whether the artist is primarily instrumental or vocal

Do NOT include:
- Biographical details (birth dates, band history, label affiliations, chart positions)
- Any copyrighted text (lyrics, liner notes, or direct quotes)
- Personal opinions or evaluative statements ("great", "influential")

If you are not confident you can accurately describe this artist, reply with exactly:
UNKNOWN_ARTIST

Output only the description (or UNKNOWN_ARTIST), nothing else."""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_sounds_like_string(raw: str) -> list[tuple[str, int | None]]:
    """Parse 'Artist A, discogs_id: 123, Artist B, discogs_id: 456' format.

    Returns list of (artist_name, discogs_id_or_None).
    """
    results = []
    segments = re.split(r",\s*discogs_id:\s*\d+", raw)
    discogs_ids = re.findall(r"discogs_id:\s*(\d+)", raw)

    for i, seg in enumerate(segments):
        artist = seg.lstrip(",").strip()
        if not artist:
            continue
        discogs_id = int(discogs_ids[i]) if i < len(discogs_ids) else None
        results.append((artist, discogs_id))

    return results


def _artist_slug(name: str) -> str:
    """Convert artist name to snake_case slug."""
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


# ---------------------------------------------------------------------------
# Catalog I/O
# ---------------------------------------------------------------------------


def load_catalog(catalog_path: Path = CATALOG_DEFAULT_PATH) -> dict:
    """Load artist catalog YAML. Returns empty dict if file is empty or absent."""
    if not catalog_path.exists():
        return {}
    with open(catalog_path) as f:
        data = yaml.safe_load(f)
    return data or {}


def _write_catalog(catalog: dict, catalog_path: Path) -> None:
    """Write catalog dict back to YAML, preserving sort order."""
    with open(catalog_path, "r") as f:
        header_lines = []
        for line in f:
            if line.startswith("#") or line.strip() == "":
                header_lines.append(line)
            else:
                break

    header = "".join(header_lines)
    body = yaml.dump(
        catalog, default_flow_style=False, sort_keys=False, allow_unicode=True
    )
    with open(catalog_path, "w") as f:
        f.write(header)
        f.write(body)


def _append_entry(
    catalog_path: Path,
    artist_name: str,
    description: Optional[str],
    style_tags: list[str],
    discogs_id: Optional[int],
    notes: str = "",
) -> None:
    """Append a new draft entry to the catalog file with chromatic_hint template."""
    slug = _artist_slug(artist_name)
    # Build the YAML body for this entry, with comment template for chromatic_hint
    # We write manually to preserve the commented-out template block.
    desc_block = ""
    if description:
        indented = "\n".join(f"    {line}" for line in description.splitlines())
        desc_block = f"  description: |\n{indented}\n"
    else:
        desc_block = "  description: null\n"

    tags_yaml = yaml.dump(style_tags, default_flow_style=True).strip()

    # Safely quote artist name if it contains special chars
    key = yaml.dump({artist_name: None}, default_flow_style=False).split(":")[0]

    discogs_str = str(discogs_id) if discogs_id is not None else "null"

    block = (
        f"\n{key}:\n"
        f"  slug: {slug}\n"
        f"  status: draft\n"
        f"{desc_block}"
        f"  style_tags: {tags_yaml}\n"
        f"  # chromatic_hint:\n"
        f"  #   temporal: past | present | future\n"
        f"  #   spatial: thing | place | person\n"
        f"  #   ontological: imagined | forgotten | known\n"
        f"  chromatic_score: null\n"
        f"  discogs_id: {discogs_str}\n"
        f'  notes: "{notes}"\n'
    )

    with open(catalog_path, "a") as f:
        f.write(block)


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def collect_sounds_like(
    thread_dir: Optional[Path] = None,
    from_training_data: bool = False,
) -> list[tuple[str, int | None]]:
    """Collect unique (artist_name, discogs_id) pairs from production plans and/or parquet.

    Deduplicates by artist name across all sources.
    """
    seen: dict[str, int | None] = {}

    if thread_dir is not None:
        thread_path = Path(thread_dir)
        for plan_path in sorted(
            (thread_path / "production").glob("*/production_plan.yml")
        ):
            with open(plan_path) as f:
                plan_data = yaml.safe_load(f) or {}
            for artist in plan_data.get("sounds_like") or []:
                name = str(artist).strip()
                if name and name not in seen:
                    seen[name] = None  # production plans don't carry discogs_id

    if from_training_data:
        try:
            import pandas as pd
        except ImportError:
            print("WARNING: pandas not available — skipping training data scan")
        else:
            if not TRAINING_PARQUET_PATH.exists():
                print(f"WARNING: Training parquet not found: {TRAINING_PARQUET_PATH}")
            else:
                df = pd.read_parquet(TRAINING_PARQUET_PATH, columns=["sounds_like"])
                for raw in df["sounds_like"].dropna().unique():
                    for artist, discogs_id in parse_sounds_like_string(str(raw)):
                        if artist and artist not in seen:
                            seen[artist] = discogs_id
                        elif (
                            artist
                            and discogs_id is not None
                            and seen.get(artist) is None
                        ):
                            seen[artist] = discogs_id

    return list(seen.items())


# ---------------------------------------------------------------------------
# Description generation
# ---------------------------------------------------------------------------


def generate_description(artist_name: str, client) -> Optional[str]:
    """Call Claude API to generate an aesthetic description for the artist.

    Returns description text or None if the artist is unknown/obscure.
    """
    prompt = _GENERATE_PROMPT_TEMPLATE.format(artist_name=artist_name)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    if text == "UNKNOWN_ARTIST" or not text:
        return None
    return text


def _infer_style_tags(description: Optional[str]) -> list[str]:
    """Return empty list — style tags are a human-curated field."""
    return []


# ---------------------------------------------------------------------------
# Generate missing entries
# ---------------------------------------------------------------------------


def generate_missing(
    artists: list[tuple[str, int | None]],
    catalog_path: Path = CATALOG_DEFAULT_PATH,
) -> dict:
    """Generate catalog entries for artists not yet in the catalog.

    Returns summary: {added: [...], already_present: [...], unknown: [...]}.
    """
    from anthropic import Anthropic

    catalog = load_catalog(catalog_path)

    to_add = [(name, did) for name, did in artists if name not in catalog]

    if not to_add:
        print("All artists already in catalog — no API calls made.")
        return {"added": [], "already_present": [n for n, _ in artists], "unknown": []}

    client = Anthropic()
    added = []
    unknown = []

    for artist_name, discogs_id in to_add:
        print(f"  Generating: {artist_name} ...", end=" ", flush=True)
        description = generate_description(artist_name, client)

        if description is None:
            notes = "Unknown artist — fill description manually"
            print("UNKNOWN")
            unknown.append(artist_name)
        else:
            notes = ""
            print("ok")

        _append_entry(
            catalog_path,
            artist_name,
            description,
            _infer_style_tags(description),
            discogs_id,
            notes=notes,
        )
        added.append(artist_name)

    already_present = [n for n, _ in artists if n not in [a for a in added]]
    print(
        f"\nAdded {len(added)} entries "
        f"({len(unknown)} unknown), "
        f"{len(already_present)} already present."
    )
    return {"added": added, "already_present": already_present, "unknown": unknown}


# ---------------------------------------------------------------------------
# Refractor scoring
# ---------------------------------------------------------------------------


def score_chromatic(
    catalog_path: Path = CATALOG_DEFAULT_PATH,
    onnx_path: Optional[str] = None,
) -> None:
    """Score all non-null descriptions through Refractor text-only mode."""
    catalog = load_catalog(catalog_path)
    if not catalog:
        print("Catalog is empty.")
        return

    try:
        from training.refractor import Refractor
    except Exception as exc:
        print(f"ERROR: Failed to import Refractor: {exc}")
        print("Use .venv312/bin/python — Refractor requires torch + numpy 1.x.")
        sys.exit(1)

    scorer = Refractor(onnx_path)
    scored = 0
    skipped = 0

    for artist_name, entry in catalog.items():
        description = entry.get("description")
        if not description:
            print(f"  SKIP (null description): {artist_name}")
            skipped += 1
            continue

        results = scorer.score_batch([{"lyric_text": description}])
        result = results[0]
        entry["chromatic_score"] = {
            "temporal": {k: round(float(v), 4) for k, v in result["temporal"].items()},
            "spatial": {k: round(float(v), 4) for k, v in result["spatial"].items()},
            "ontological": {
                k: round(float(v), 4) for k, v in result["ontological"].items()
            },
            "confidence": round(float(result["confidence"]), 4),
        }
        scored += 1
        print(f"  Scored: {artist_name}")

    _write_catalog(catalog, catalog_path)
    print(f"\nScored {scored} entries, skipped {skipped} (null description).")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(catalog_path: Path = CATALOG_DEFAULT_PATH) -> None:
    """Print catalog status counts."""
    catalog = load_catalog(catalog_path)
    total = len(catalog)
    n_draft = sum(1 for e in catalog.values() if e.get("status") == "draft")
    n_reviewed = sum(1 for e in catalog.values() if e.get("status") == "reviewed")
    n_hint = sum(1 for e in catalog.values() if e.get("chromatic_hint"))
    n_score = sum(1 for e in catalog.values() if e.get("chromatic_score"))
    n_null_desc = sum(1 for e in catalog.values() if not e.get("description"))

    print(f"\nArtist Catalog — {catalog_path}")
    print(f"  Total artists:        {total}")
    print(f"  Status draft:         {n_draft}")
    print(f"  Status reviewed:      {n_reviewed}")
    print(f"  Null description:     {n_null_desc}  (needs manual fill)")
    print(f"  chromatic_hint filled:{n_hint}")
    print(f"  chromatic_score set:  {n_score}")


# ---------------------------------------------------------------------------
# Pipeline utility — shared with lyric/chord pipelines
# ---------------------------------------------------------------------------


def load_artist_context(
    sounds_like: list[str],
    catalog_path: Path = CATALOG_DEFAULT_PATH,
) -> str:
    """Return a formatted STYLE REFERENCES block for the given artist names.

    Prefers reviewed entries over draft. Prints a note for missing artists.
    Returns an empty string when no matches are found.
    """
    if not sounds_like:
        return ""
    catalog = load_catalog(catalog_path)

    blocks: list[str] = []
    for artist_name in sounds_like:
        entry = catalog.get(artist_name)
        if entry is None:
            print(
                f"  Artist '{artist_name}' not in catalog — "
                "run artist_catalog.py --generate-missing to add"
            )
            continue

        description = entry.get("description")
        if not description:
            continue

        status = entry.get("status", "draft")
        if status != "reviewed":
            print(f"  Using draft description for '{artist_name}' — consider reviewing")

        blocks.append(f"- {artist_name}: {description.strip()}")

    if not blocks:
        return ""

    lines = [
        "STYLE REFERENCES (aesthetic context only):",
        "Use these as aesthetic reference only — do not imitate specific lyrics "
        "or identifiable phrases.",
        "",
    ] + blocks

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Artist Style Catalog — generate and manage aesthetic descriptions"
    )
    parser.add_argument(
        "--thread",
        help="Thread directory to scan for sounds_like entries in production plans",
    )
    parser.add_argument(
        "--from-training-data",
        action="store_true",
        help="Also scan training_data_with_embeddings.parquet for sounds_like",
    )
    parser.add_argument(
        "--generate-missing",
        action="store_true",
        help="Generate descriptions for uncatalogued artists via Claude API",
    )
    parser.add_argument(
        "--score-chromatic",
        action="store_true",
        help="Score all non-null descriptions through Refractor (.venv312)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print catalog status summary",
    )
    parser.add_argument(
        "--catalog",
        default=str(CATALOG_DEFAULT_PATH),
        help=f"Path to artist_catalog.yml (default: {CATALOG_DEFAULT_PATH})",
    )
    parser.add_argument(
        "--onnx-path",
        help="Path to refractor.onnx (for --score-chromatic)",
    )
    args = parser.parse_args()

    catalog_path = Path(args.catalog)

    if args.generate_missing:
        if not args.thread and not args.from_training_data:
            print(
                "ERROR: --generate-missing requires --thread and/or --from-training-data"
            )
            sys.exit(1)
        print("Collecting sounds_like artists...")
        artists = collect_sounds_like(
            thread_dir=Path(args.thread) if args.thread else None,
            from_training_data=args.from_training_data,
        )
        print(f"Found {len(artists)} unique artists.")
        generate_missing(artists, catalog_path)

    if args.score_chromatic:
        score_chromatic(catalog_path, onnx_path=args.onnx_path)

    if args.summary or not (args.generate_missing or args.score_chromatic):
        print_summary(catalog_path)


if __name__ == "__main__":
    main()
