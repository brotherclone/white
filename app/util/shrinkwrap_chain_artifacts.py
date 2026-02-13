"""Shrink-wrap chain artifact threads into a clean output directory.

Reads from chain_artifacts/ (untouched) and writes clean copies to an output directory:
- Copies non-debug files with clean directory names ({color}-{slugified-title})
- Skips debug files (rebracketing analyses, traces, etc.) unless --archive is set
- Generates manifest.yml per thread and index.yml at top level

Usage:
    python -m app.util.shrinkwrap_chain_artifacts                              # output to shrinkwrapped/
    python -m app.util.shrinkwrap_chain_artifacts --output-dir my_output       # custom output dir
    python -m app.util.shrinkwrap_chain_artifacts --dry-run                    # preview changes
    python -m app.util.shrinkwrap_chain_artifacts --archive                    # include debug in .debug/
    python -m app.util.shrinkwrap_chain_artifacts --thread <uuid>              # process one thread
"""

import argparse
import logging
import re
import shutil
import uuid
import yaml

from pathlib import Path
from typing import Optional

from app.util.generate_negative_constraints import (
    generate_constraints,
    write_constraints,
)

logger = logging.getLogger(__name__)

# Debug file patterns in md/ — these are intermediate White Agent outputs
DEBUG_FILE_PATTERNS = [
    r"white_agent_.*_rebracketing_analysis\.(?:md|json)$",
    r"white_agent_.*_document_synthesis\.(?:md|json)$",
    r"white_agent_.*_META_REBRACKETING\.(?:md|json)$",
    # CHROMATIC_SYNTHESIS removed from debug patterns — keep it as a valuable output
    r"white_agent_.*_facet_evolution\.(?:md|json)$",
    r"white_agent_.*_transformation_traces\.(?:md|json)$",
    # Broader fallback patterns to catch other analysis/trace variants
    r".*_analysis\.(?:md|json)$",
    r".*trace.*\.(?:md|json)$",
]

# EVP intermediate file patterns — legacy segment/blended audio that should be stripped
EVP_INTERMEDIATE_PATTERNS = [
    r".*_segment_\d+\.wav$",
    r"blended.*\.wav$",
]


def is_uuid(name: str) -> bool:
    """Check if a string is a valid UUID."""
    try:
        uuid.UUID(name)
        return True
    except ValueError:
        return False


def slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"['\u2018\u2019\u201B]", "", text)  # Remove apostrophes
    text = re.sub(r"[^a-z0-9]+", "-", text)  # Replace non-alphanumeric with hyphens
    text = re.sub(r"-+", "-", text)  # Collapse multiple hyphens
    text = text.strip("-")  # Remove leading/trailing hyphens
    return text[:80]  # Cap length


def is_debug_file(filename: str) -> bool:
    """Check if a file is a debug/intermediate output.

    Matching is case-insensitive to catch variations in filename casing.
    """
    for pattern in DEBUG_FILE_PATTERNS:
        if re.search(pattern, filename, flags=re.IGNORECASE):
            return True
    return False


def is_evp_intermediate(filename: str) -> bool:
    """Check if a file is a legacy EVP intermediate (segment/blended audio).

    Matching is case-insensitive.
    """
    for pattern in EVP_INTERMEDIATE_PATTERNS:
        if re.search(pattern, filename, flags=re.IGNORECASE):
            return True
    return False


def parse_thread(thread_dir: Path) -> Optional[dict]:
    """Parse a thread directory and extract the final song proposal metadata.

    Returns:
        Dict with thread metadata, or None if unparseable.
    """
    thread_id = thread_dir.name

    # Find all_song_proposals YAML
    proposal_files = list(thread_dir.glob("yml/all_song_proposals_*.yml"))
    if not proposal_files:
        logger.warning(f"No song proposals found in {thread_id}")
        return None

    proposal_path = proposal_files[0]

    try:
        with open(proposal_path) as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to parse {proposal_path}: {e}")
        return None

    iterations = data.get("iterations", [])
    if not iterations:
        logger.warning(f"No iterations in {thread_id}")
        return None

    # The last iteration is the final synthesis
    final = iterations[-1]

    rainbow_color = final.get("rainbow_color", {})
    color_name = rainbow_color.get("color_name", "unknown")

    return {
        "thread_id": thread_id,
        "title": final.get("title", "untitled"),
        "bpm": final.get("bpm"),
        "key": final.get("key"),
        "tempo": final.get("tempo"),
        "concept": final.get("concept", ""),
        "rainbow_color": color_name,
        "mnemonic": rainbow_color.get("mnemonic_character_value", "?"),
        "mood": final.get("mood", []),
        "genres": final.get("genres", []),
        "agent_name": final.get("agent_name", ""),
        "iteration_count": len(iterations),
        "timestamp": final.get("timestamp"),
    }


def find_debug_files(thread_dir: Path) -> list[Path]:
    """Find all debug/intermediate files in a thread directory.

    Debug files can be .md or .json and may appear outside of `md/`.
    Search the entire thread tree to locate them.
    """
    debug_files = []
    for f in thread_dir.rglob("*"):
        if f.is_file() and is_debug_file(f.name):
            debug_files.append(f)
    return sorted(debug_files)


def generate_directory_name(metadata: dict, existing_names: set[str]) -> str:
    """Generate a clean directory name from metadata."""
    color = slugify(metadata["rainbow_color"])
    title = slugify(metadata["title"])
    base_name = f"{color}-{title}" if title else color

    # Handle collisions
    name = base_name
    counter = 2
    while name in existing_names:
        name = f"{base_name}-{counter}"
        counter += 1

    return name


def write_manifest(output_dir: Path, metadata: dict) -> Path:
    """Write manifest.yml into the output directory."""
    manifest_path = output_dir / "manifest.yml"

    manifest = {
        "title": metadata["title"],
        "bpm": metadata["bpm"],
        "key": metadata["key"],
        "tempo": metadata["tempo"],
        "concept": metadata["concept"],
        "rainbow_color": metadata["rainbow_color"],
        "mnemonic": metadata["mnemonic"],
        "mood": metadata["mood"],
        "genres": metadata["genres"],
        "agent_name": metadata["agent_name"],
        "iteration_count": metadata["iteration_count"],
        "thread_id": metadata["thread_id"],
        "timestamp": metadata["timestamp"],
    }

    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, allow_unicode=True, width=120)

    return manifest_path


def write_index(output_dir: Path, all_metadata: list[dict]) -> Path:
    """Write the top-level index.yml."""
    index_path = output_dir / "index.yml"

    threads = []
    for meta in sorted(all_metadata, key=lambda m: m.get("title", "")):
        entry = {
            "thread_id": meta["thread_id"],
            "directory": meta.get("directory_name", meta["thread_id"]),
            "title": meta["title"],
            "bpm": meta["bpm"],
            "key": meta["key"],
            "rainbow_color": meta["rainbow_color"],
            "concept": (
                meta["concept"][:200] + "..."
                if len(meta.get("concept", "")) > 200
                else meta.get("concept", "")
            ),
            "iteration_count": meta["iteration_count"],
        }
        threads.append(entry)

    index = {
        "thread_count": len(threads),
        "threads": threads,
    }

    with open(index_path, "w") as f:
        yaml.dump(index, f, default_flow_style=False, allow_unicode=True, width=120)

    return index_path


def copy_thread_files(
    source_dir: Path,
    dest_dir: Path,
    include_debug: bool = False,
) -> dict:
    """Copy non-debug files from source thread to destination.

    This function avoids creating destination subdirectories unless files
    are actually copied into them. Debug files are only written into a
    `.debug/` subdirectory if `include_debug` is True and at least one
    debug file is being archived.

    Returns:
        Dict with file counts: copied, skipped_debug, skipped_evp.
    """
    copied = 0
    skipped_debug = 0
    skipped_evp = 0

    # Track whether we've created the .debug directory yet
    debug_dest = None

    for subdir in sorted(source_dir.iterdir()):
        if not subdir.is_dir():
            continue

        # We'll only create this when we need to copy a non-debug file into it
        dest_subdir = dest_dir / subdir.name
        dest_subdir_created = False

        for f in sorted(subdir.iterdir()):
            if not f.is_file():
                continue
            if f.name == ".DS_Store":
                continue

            if is_evp_intermediate(f.name):
                skipped_evp += 1
                continue

            if is_debug_file(f.name):
                if include_debug:
                    # Lazily create .debug/ only when the first debug file is copied
                    if debug_dest is None:
                        debug_dest = dest_dir / ".debug"
                        debug_dest.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(f), str(debug_dest / f.name))
                    copied += 1
                else:
                    skipped_debug += 1
                continue

            # Non-debug, non-evp file: ensure dest_subdir exists then copy
            if not dest_subdir_created:
                dest_subdir.mkdir(parents=True, exist_ok=True)
                dest_subdir_created = True

            shutil.copy2(str(f), str(dest_subdir / f.name))
            copied += 1

    return {
        "copied": copied,
        "skipped_debug": skipped_debug,
        "skipped_evp": skipped_evp,
    }


def shrinkwrap_thread(
    thread_dir: Path,
    output_dir: Path,
    existing_names: set[str],
    dry_run: bool = False,
    archive: bool = False,
) -> Optional[dict]:
    """Shrink-wrap a single thread into the output directory.

    Returns:
        Metadata dict with directory_name added, or None on failure.
    """
    thread_id = thread_dir.name

    # Skip non-UUID directories
    if not is_uuid(thread_id):
        logger.debug(f"Skipping {thread_id} (not a UUID)")
        return None

    metadata = parse_thread(thread_dir)
    if not metadata:
        return None

    new_name = generate_directory_name(metadata, existing_names)
    metadata["directory_name"] = new_name
    debug_files = find_debug_files(thread_dir)

    if dry_run:
        print(f"\n  {thread_id}")
        print(f"  → {new_name}/")
        print(f"    title: {metadata['title']}")
        print(
            f"    color: {metadata['rainbow_color']}, key: {metadata['key']}, bpm: {metadata['bpm']}"
        )
        print(f"    iterations: {metadata['iteration_count']}")
        if debug_files:
            print(
                f"    debug files to {'archive' if archive else 'skip'}: {len(debug_files)}"
            )
        existing_names.add(new_name)
        return metadata

    # Create output directory
    dest_dir = output_dir / new_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Copy non-debug files
    counts = copy_thread_files(thread_dir, dest_dir, include_debug=archive)
    logger.info(
        f"{thread_id} → {new_name}: "
        f"{counts['copied']} files copied, {counts['skipped_debug']} debug skipped, "
        f"{counts['skipped_evp']} evp intermediates skipped"
    )

    # Write manifest
    write_manifest(dest_dir, metadata)

    existing_names.add(new_name)
    return metadata


def load_orphaned_manifests(output_dir: Path, known_dirs: set[str]) -> list[dict]:
    """Load manifests from output directories not already tracked.

    When chain_artifacts are deleted or become unparseable, the shrinkwrapped
    output directories (with their manifests) are the only remaining record.
    This scans for those orphaned directories so the index stays complete.
    """
    orphaned = []
    if not output_dir.exists():
        return orphaned

    for d in sorted(output_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        if d.name in known_dirs:
            continue

        manifest_path = d / "manifest.yml"
        if not manifest_path.exists():
            continue

        try:
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f)
            if not manifest or not manifest.get("title"):
                continue

            metadata = {
                "thread_id": manifest.get("thread_id", "unknown"),
                "title": manifest["title"],
                "bpm": manifest.get("bpm"),
                "key": manifest.get("key"),
                "tempo": manifest.get("tempo"),
                "concept": manifest.get("concept", ""),
                "rainbow_color": manifest.get("rainbow_color", "unknown"),
                "mnemonic": manifest.get("mnemonic", "?"),
                "mood": manifest.get("mood", []),
                "genres": manifest.get("genres", []),
                "agent_name": manifest.get("agent_name", ""),
                "iteration_count": manifest.get("iteration_count", 0),
                "timestamp": manifest.get("timestamp"),
                "directory_name": d.name,
            }
            orphaned.append(metadata)
            logger.info(f"Recovered orphaned manifest: {d.name}")
        except Exception as e:
            logger.warning(f"Failed to load manifest from {d.name}: {e}")

    return orphaned


def shrinkwrap(
    artifacts_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
    archive: bool = False,
    thread_filter: Optional[str] = None,
) -> dict:
    """Shrink-wrap all (or one) thread(s) into the output directory.

    Args:
        artifacts_dir: Source chain_artifacts/ directory (not modified).
        output_dir: Destination directory for clean output.
        dry_run: Preview changes without writing.
        archive: Include debug files in .debug/ subdirectory.
        thread_filter: Process only this thread UUID.

    Returns:
        Summary dict with counts and metadata list.
    """
    if not artifacts_dir.exists():
        logger.error(f"Artifacts directory not found: {artifacts_dir}")
        return {"processed": 0, "skipped": 0, "failed": 0}

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Collect existing output directory names to avoid collisions
    existing_names: set[str] = set()
    if output_dir.exists():
        existing_names = {d.name for d in output_dir.iterdir() if d.is_dir()}

    all_metadata = []
    processed = 0
    failed = 0

    thread_dirs = sorted(artifacts_dir.iterdir())
    for thread_dir in thread_dirs:
        if not thread_dir.is_dir():
            continue
        if not is_uuid(thread_dir.name):
            continue
        if thread_filter and thread_dir.name != thread_filter:
            continue

        # Skip if already in output
        metadata_check = parse_thread(thread_dir)
        if metadata_check:
            candidate_name = generate_directory_name(metadata_check, set())
            if candidate_name in existing_names and not dry_run:
                logger.info(
                    f"Skipping {thread_dir.name} (already in output as {candidate_name})"
                )
                metadata_check["directory_name"] = candidate_name
                all_metadata.append(metadata_check)
                continue

        metadata = shrinkwrap_thread(
            thread_dir,
            output_dir,
            existing_names,
            dry_run=dry_run,
            archive=archive,
        )

        if metadata:
            all_metadata.append(metadata)
            processed += 1
        else:
            failed += 1

    # Recover orphaned manifests from output directories whose chain_artifacts
    # are no longer parseable (deleted, reorganized, etc.)
    tracked_dirs = {
        m.get("directory_name") for m in all_metadata if m.get("directory_name")
    }
    orphaned = load_orphaned_manifests(output_dir, tracked_dirs)
    if orphaned:
        all_metadata.extend(orphaned)
        if not dry_run:
            logger.info(f"Recovered {len(orphaned)} orphaned manifests")
        else:
            print(
                f"\n  + {len(orphaned)} existing directories with manifests (from previous runs)"
            )

    if all_metadata and not dry_run:
        index_path = write_index(output_dir, all_metadata)
        logger.info(f"Wrote index.yml ({len(all_metadata)} threads) in {index_path}")
        constraints = generate_constraints(index_path)
        constraints_path = output_dir / "negative_constraints.yml"
        write_constraints(constraints_path, constraints)
        logger.info(f"Auto-regenerated constraints in {constraints_path}")

    if dry_run and all_metadata:
        print(f"\nSummary: {processed} threads to process, {failed} failed")

    return {
        "processed": processed,
        "failed": failed,
        "metadata": all_metadata,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Shrink-wrap chain artifact threads into a clean output directory"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("chain_artifacts"),
        help="Source chain_artifacts directory (not modified)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("shrinkwrapped"),
        help="Output directory for clean artifacts (default: shrinkwrapped/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without writing files"
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Include debug files in .debug/ subdirectory",
    )
    parser.add_argument("--thread", type=str, help="Process a single thread by UUID")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.dry_run:
        print("DRY RUN — no files will be written\n")

    result = shrinkwrap(
        artifacts_dir=args.artifacts_dir,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        archive=args.archive,
        thread_filter=args.thread,
    )

    print(f"\nDone: {result['processed']} processed, {result['failed']} failed")


if __name__ == "__main__":
    main()
