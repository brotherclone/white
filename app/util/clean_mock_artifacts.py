#!/usr/bin/env python3
"""Utility to clean up mock artifacts under the repository's `chain_artifacts` folder.

Usage examples:
  python -m app.util.clean_mock_artifacts --dry-run
  python app/util/clean_mock_artifacts.py --yes

Features:
- Defaults to <repo_root>/chain_artifacts
- Matches directory names that contain 'mock' (case-insensitive) or are named 'UNKNOWN_THREAD_ID'
- Supports --dry-run to list candidates without deleting
- Requires interactive confirmation unless --yes is provided
- Safe deletion with exception handling and summary
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List


def find_mock_dirs(base_dir: Path) -> List[Path]:
    """Return a list of directories under base_dir whose name contains 'mock' (case-insensitive) or is 'UNKNOWN_THREAD_ID'."""
    if not base_dir.exists():
        return []
    results: List[Path] = []
    for entry in base_dir.iterdir():
        # only top-level entries (folders) are considered
        try:
            if entry.is_dir() and (
                "mock" in entry.name.lower()
                or entry.name == "UNKNOWN_THREAD_ID"
                or entry.name == "test_thread"
            ):
                results.append(entry)
        except OSError:
            # skip entries we can't stat
            continue
    return sorted(results)


def delete_paths(paths: List[Path], verbose: bool = False) -> int:
    """Delete the given paths. Returns number of successfully deleted entries."""
    deleted = 0
    for p in paths:
        try:
            if p.is_symlink() or p.is_file():
                if verbose:
                    print(f"Removing file/symlink: {p}")
                p.unlink()
            elif p.is_dir():
                if verbose:
                    print(f"Removing directory tree: {p}")
                shutil.rmtree(p)
            else:
                # fallback: try rmtree
                if verbose:
                    print(f"Removing unknown path type with rmtree: {p}")
                shutil.rmtree(p)
            deleted += 1
        except Exception as e:
            print(f"Failed to remove {p}: {e}", file=sys.stderr)
    return deleted


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Delete directories in <repo_root>/chain_artifacts whose names contain 'mock' or are named 'UNKNOWN_THREAD_ID' or test_thread",
    )
    p.add_argument(
        "--base",
        type=Path,
        default=None,
        help="Base directory to scan (defaults to repository_root/chain_artifacts)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching directories but don't delete",
    )
    p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Don't prompt; delete matching directories immediately",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    # resolve repo root relative to this file: ../../ (app/util -> app -> repo root)
    repo_root = Path(__file__).resolve().parents[2]
    base_dir = args.base or (repo_root / "chain_artifacts")

    if not base_dir.exists():
        print(f"Base directory does not exist: {base_dir}")
        return 1

    candidates = find_mock_dirs(base_dir)
    if not candidates:
        print(f"No directories containing 'mock' found under: {base_dir}")
        return 0

    print("Found the following candidate directories to remove:")
    for p in candidates:
        print("  -", p)

    if args.dry_run:
        print("Dry run: no files were deleted.")
        return 0

    if not args.yes:
        try:
            resp = input("Proceed to delete these directories? [y/N]: ")
        except EOFError:
            resp = "n"
        if resp.strip().lower() not in ("y", "yes"):
            print("Aborted by user.")
            return 2

    deleted = delete_paths(candidates, verbose=args.verbose)
    print(f"Deleted {deleted} of {len(candidates)} candidate(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
