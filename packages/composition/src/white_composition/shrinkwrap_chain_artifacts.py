"""Shrink-wrap chain artifact threads into a clean output directory.

Reads from chain_artifacts/ (untouched) and writes clean copies to an output directory:
- Copies non-debug files with clean directory names ({color}-{slugified-title})
- Skips debug files (rebracketing analyses, traces, etc.) unless --archive is set
- Generates manifest.yml per thread and index.yml at top level

Usage:
    python -m app.util.shrinkwrap_chain_artifacts                              # output to shrink_wrapped/
    python -m app.util.shrinkwrap_chain_artifacts --output-dir my_output       # custom output dir
    python -m app.util.shrinkwrap_chain_artifacts --dry-run                    # preview changes
    python -m app.util.shrinkwrap_chain_artifacts --archive                    # include debug in .debug/
    python -m app.util.shrinkwrap_chain_artifacts --thread <uuid>              # process one thread
"""

import argparse
import logging
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Optional

import yaml

from white_extraction.util.generate_negative_constraints import (
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


# Compiled patterns for clean_filename(), applied in order.
_UUID_PREFIX_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_[a-zA-Z]_(.+)$",
    re.IGNORECASE,
)
_WHITE_AGENT_RE = re.compile(
    r"^white_agent_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_(.+)$",
    re.IGNORECASE,
)
_ALL_PROPOSALS_RE = re.compile(
    r"^(all_song_proposals)_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(\..+)$",
    re.IGNORECASE,
)
_SONG_PROPOSAL_RE = re.compile(
    r"^song_proposal_.+?_(.+)$",
    re.IGNORECASE,
)


def clean_filename(raw_name: str) -> str:
    """Return a human-readable filename by stripping UUID/agent/color prefixes.

    Rules (applied in order; first match wins):
    1. <uuid>_<char>_<name>.<ext>              → <name>.<ext>
    2. white_agent_<thread-uuid>_<TYPE>.<ext>  → <type_lowercase>.<ext>
    3. all_song_proposals_<thread-uuid>.<ext>  → all_song_proposals.<ext>
    4. song_proposal_<Color...>_<name>.<ext>   → <name>.<ext>
    5. No match                                → unchanged
    """
    m = _UUID_PREFIX_RE.match(raw_name)
    if m:
        return m.group(1)

    m = _WHITE_AGENT_RE.match(raw_name)
    if m:
        return m.group(1).lower()

    m = _ALL_PROPOSALS_RE.match(raw_name)
    if m:
        return f"{m.group(1)}{m.group(2)}"

    m = _SONG_PROPOSAL_RE.match(raw_name)
    if m:
        return m.group(1)

    return raw_name


def resolve_collision(clean_name: str, used: set[str]) -> str:
    """Append _2, _3, … before the extension until the name is unique in *used*."""
    if clean_name not in used:
        return clean_name
    p = Path(clean_name)
    stem, suffix = p.stem, p.suffix
    counter = 2
    while True:
        candidate = f"{stem}_{counter}{suffix}"
        if candidate not in used:
            return candidate
        counter += 1


def rewrite_file_name_field(file_path: Path, clean_name: str) -> None:
    """Rewrite a bare ``file_name:`` line in a copied file to *clean_name*.

    Uses line-level replacement to avoid YAML round-trip issues with Python
    object tags that appear in some chain artifact files.  Best-effort: any
    read/write error is silently ignored so the file is still usable.
    """
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception:
        return

    new_lines = []
    replaced = False
    for line in text.splitlines(keepends=True):
        if not replaced and re.match(r"^file_name:\s+", line):
            new_lines.append(f"file_name: {clean_name}\n")
            replaced = True
        else:
            new_lines.append(line)

    if replaced:
        try:
            file_path.write_text("".join(new_lines), encoding="utf-8")
        except Exception:
            pass


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


_ACCIDENTAL_MAP = {"sharp": "#", "flat": "b"}


def _flatten_key_dict(d: dict) -> str:
    """Flatten a KeySignature dict to a 'tonic mode' string.

    Supports two shapes:
      - KeySignature.model_dump(mode="json"):
          {"note": {"pitch_name": "F", "accidental": "sharp"}, "mode": {"name": "minor"}}
      - Legacy flat shape: {"tonic": "F#", "mode": "minor"}
    """
    if "note" in d:
        note = d.get("note") or {}
        pitch = note.get("pitch_name", "") if isinstance(note, dict) else ""
        acc = note.get("accidental", "") if isinstance(note, dict) else ""
        tonic = f"{pitch}{_ACCIDENTAL_MAP.get(acc, '')}" if pitch else ""
    else:
        tonic = d.get("tonic", "")

    mode_val = d.get("mode", "")
    mode_str = mode_val.get("name", "") if isinstance(mode_val, dict) else str(mode_val)

    return f"{tonic} {mode_str}".strip() if tonic else "unknown"


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
    if isinstance(rainbow_color, dict):
        color_name = rainbow_color.get("color_name", "unknown")
        mnemonic = rainbow_color.get("mnemonic_character_value", "?")
    else:
        color_name = str(rainbow_color) if rainbow_color else "unknown"
        mnemonic = "?"

    raw_key = final.get("key")
    if isinstance(raw_key, dict):
        key_str = _flatten_key_dict(raw_key)
    else:
        key_str = str(raw_key) if raw_key else "unknown"

    return {
        "thread_id": thread_id,
        "title": final.get("title", "untitled"),
        "bpm": final.get("bpm"),
        "key": key_str,
        "tempo": final.get("tempo"),
        "concept": final.get("concept", ""),
        "rainbow_color": color_name,
        "mnemonic": mnemonic,
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
    """Copy non-debug files from source thread to destination with clean names.

    Each output filename is run through ``clean_filename()`` to strip UUID
    prefixes, color-char codes, and agent/thread-name prefixes.  Collisions
    within the same subdirectory are resolved by appending ``_2``, ``_3``, …
    before the extension.  After copying, any ``file_name:`` field inside the
    file is rewritten to the clean name.

    Debug files are only written into a ``.debug/`` subdirectory when
    ``include_debug`` is True.  Destination subdirectories are created lazily
    (only when at least one file lands there).

    Returns:
        Dict with file counts: copied, skipped_debug, skipped_evp.
    """
    copied = 0
    skipped_debug = 0
    skipped_evp = 0

    debug_dest = None
    # Per-subdirectory sets of already-used clean names for collision detection.
    used_names: dict[str, set[str]] = {}

    for subdir in sorted(source_dir.iterdir()):
        if not subdir.is_dir():
            continue

        # Always skip the debug/ subdirectory — state snapshots are never useful
        # in the shrinkwrapped output.  Use --archive for debug inclusion.
        if subdir.name == "debug":
            continue

        dest_subdir = dest_dir / subdir.name
        dest_subdir_created = False
        subdir_used = used_names.setdefault(subdir.name, set())

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
                    if debug_dest is None:
                        debug_dest = dest_dir / ".debug"
                        debug_dest.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(f), str(debug_dest / f.name))
                    copied += 1
                else:
                    skipped_debug += 1
                continue

            # Non-debug, non-evp: clean the name, resolve collisions, then copy.
            if not dest_subdir_created:
                dest_subdir.mkdir(parents=True, exist_ok=True)
                dest_subdir_created = True

            clean_name = clean_filename(f.name)
            clean_name = resolve_collision(clean_name, subdir_used)
            subdir_used.add(clean_name)

            dest_path = dest_subdir / clean_name
            shutil.copy2(str(f), str(dest_path))
            rewrite_file_name_field(dest_path, clean_name)
            copied += 1

    return {
        "copied": copied,
        "skipped_debug": skipped_debug,
        "skipped_evp": skipped_evp,
    }


_SKIP_PROPOSALS = {"evp.yml", "all_song_proposals.yml"}
_PROPOSAL_REQUIRED_KEYS = {"bpm", "key", "rainbow_color"}


def scaffold_song_productions(
    thread_dest_dir: Path, yml_dir: Path, force: bool = False
) -> list[str]:
    """Create production/<slug>/manifest_bootstrap.yml for each song proposal in yml_dir.

    A YAML file is treated as a song proposal if it contains bpm, key, and rainbow_color.
    Known non-proposal files (evp.yml, all_song_proposals.yml) are always skipped.
    Existing manifest_bootstrap.yml files are never overwritten (idempotent).

    Returns list of production slugs created (or already existing).
    """
    if not yml_dir.exists():
        return []

    created = []
    for yml_path in sorted(yml_dir.glob("*.yml")):
        if yml_path.name in _SKIP_PROPOSALS:
            continue
        try:
            with open(yml_path) as f:
                proposal = yaml.safe_load(f)
        except Exception:
            continue
        if not isinstance(proposal, dict):
            continue
        if not _PROPOSAL_REQUIRED_KEYS.issubset(proposal):
            continue

        slug = yml_path.stem
        prod_dir = thread_dest_dir / "production" / slug
        manifest_path = prod_dir / "manifest_bootstrap.yml"
        if manifest_path.exists() and not force:
            continue

        prod_dir.mkdir(parents=True, exist_ok=True)
        rc = proposal.get("rainbow_color")
        rainbow_color = rc.get("color_name") if isinstance(rc, dict) else rc
        bootstrap = {
            "title": proposal.get("title") or slug,
            "key": proposal.get("key"),
            "bpm": proposal.get("bpm"),
            "rainbow_color": rainbow_color,
            "singer": proposal.get("singer") or None,
        }
        with open(manifest_path, "w") as f:
            yaml.dump(
                bootstrap, f, allow_unicode=True, sort_keys=False, width=float("inf")
            )
        created.append(slug)

    return created


def shrinkwrap_thread(
    thread_dir: Path,
    output_dir: Path,
    existing_names: set[str],
    dry_run: bool = False,
    archive: bool = False,
    scaffold: bool = True,
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
        if scaffold:
            yml_dir = thread_dir / "yml"
            if yml_dir.exists():
                proposals = []
                for p in sorted(yml_dir.glob("*.yml")):
                    if p.name in _SKIP_PROPOSALS:
                        continue
                    try:
                        with open(p) as _f:
                            _data = yaml.safe_load(_f)
                        if isinstance(_data, dict) and _PROPOSAL_REQUIRED_KEYS.issubset(
                            _data
                        ):
                            proposals.append(p.stem)
                    except Exception:
                        pass
                if proposals:
                    print(f"    production dirs to scaffold: {proposals}")
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

    # Scaffold production directories from song proposals
    if scaffold:
        slugs = scaffold_song_productions(dest_dir, dest_dir / "yml")
        if slugs:
            logger.info(f"  scaffolded production dirs: {slugs}")

    existing_names.add(new_name)
    return metadata


def load_orphaned_manifests(output_dir: Path, known_dirs: set[str]) -> list[dict]:
    """Load manifests from output directories not already tracked.

    When chain_artifacts are deleted or become unparseable, the shrink_wrapped
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
    scaffold: bool = True,
    delete_incomplete: bool = True,
) -> dict:
    """Shrink-wrap all (or one) thread(s) into the output directory.

    Args:
        artifacts_dir: Source chain_artifacts/ directory. When delete_incomplete
            is True, incomplete thread directories are deleted from this directory.
        output_dir: Destination directory for clean output.
        dry_run: Preview changes without writing.
        archive: Include debug files in .debug/ subdirectory.
        thread_filter: Process only this thread UUID.
        delete_incomplete: Delete thread dirs that lack a run_success sentinel
            AND are unparseable. Legacy threads without a sentinel but with valid
            proposals are processed normally.

    Returns:
        Summary dict with counts and metadata list.
    """
    if not artifacts_dir.exists():
        logger.error(f"Artifacts directory not found: {artifacts_dir}")
        return {"processed": 0, "failed": 0, "deleted": 0, "metadata": []}

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Collect existing output directory names to avoid collisions
    existing_names: set[str] = set()
    if output_dir.exists():
        existing_names = {d.name for d in output_dir.iterdir() if d.is_dir()}

    all_metadata = []
    processed = 0
    failed = 0
    deleted = 0

    thread_dirs = sorted(artifacts_dir.iterdir())
    for thread_dir in thread_dirs:
        if not thread_dir.is_dir():
            continue
        if not is_uuid(thread_dir.name):
            continue
        if thread_filter and thread_dir.name != thread_filter:
            continue

        # Threads without a run_success sentinel are either incomplete/crashed or
        # legacy (pre-sentinel).  Parse first: a parseable thread is legacy and
        # processed normally; an unparseable one is treated as incomplete.
        sentinel = thread_dir / "run_success"
        metadata_check = None
        if not sentinel.exists():
            metadata_check = parse_thread(thread_dir)
            if metadata_check:
                logger.info(
                    f"Processing legacy thread {thread_dir.name} (no run_success sentinel)"
                )
            else:
                if dry_run:
                    print(
                        f"  [incomplete] {thread_dir.name} — no run_success sentinel and unparseable"
                    )
                elif delete_incomplete:
                    logger.info(
                        f"Deleting incomplete thread {thread_dir.name} "
                        f"(no run_success sentinel and unparseable)"
                    )
                    try:
                        shutil.rmtree(thread_dir)
                        deleted += 1
                    except OSError:
                        logger.warning(
                            f"Failed to delete incomplete thread {thread_dir.name}",
                            exc_info=True,
                        )
                        failed += 1
                else:
                    logger.debug(f"Skipping incomplete thread {thread_dir.name}")
                continue

        # Skip if already in output
        if metadata_check is None:
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
            scaffold=scaffold,
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
        "deleted": deleted,
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
        default=Path(os.getenv("SHRINKWRAP_OUTPUT_DIR", "shrink_wrapped")),
        help="Output directory for clean artifacts (default: $SHRINKWRAP_OUTPUT_DIR or shrink_wrapped/)",
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
    parser.add_argument(
        "--no-delete-incomplete",
        action="store_true",
        help="Skip deletion of incomplete (crashed/failed) thread directories",
    )
    parser.add_argument(
        "--no-scaffold",
        action="store_true",
        help="Skip scaffolding production directories from song proposals",
    )
    parser.add_argument(
        "--scaffold-only",
        action="store_true",
        help="Only scaffold production dirs in existing output dirs, then exit (backfill mode)",
    )
    parser.add_argument(
        "--force-scaffold",
        action="store_true",
        help="Overwrite existing manifest_bootstrap.yml files during scaffolding",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.scaffold_only:
        output_dir = args.output_dir
        total = 0
        for thread_dir in sorted(output_dir.iterdir()):
            if not thread_dir.is_dir() or thread_dir.name.startswith("."):
                continue
            yml_dir = thread_dir / "yml"
            if not yml_dir.exists():
                continue
            slugs = scaffold_song_productions(
                thread_dir, yml_dir, force=args.force_scaffold
            )
            if slugs:
                print(f"  {thread_dir.name}: {slugs}")
                total += len(slugs)
        print(f"\nDone: {total} production dirs scaffolded")
        return

    if args.dry_run:
        print("DRY RUN — no files will be written\n")

    result = shrinkwrap(
        artifacts_dir=args.artifacts_dir,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        archive=args.archive,
        thread_filter=args.thread,
        scaffold=not args.no_scaffold,
        delete_incomplete=not args.no_delete_incomplete,
    )

    print(
        f"\nDone: {result['processed']} processed, {result['failed']} failed, "
        f"{result.get('deleted', 0)} incomplete deleted"
    )


if __name__ == "__main__":
    main()
