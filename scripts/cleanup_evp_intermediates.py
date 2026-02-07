"""Remove legacy EVP intermediate audio files (segments and blended).

Usage:
    python scripts/cleanup_evp_intermediates.py --dry-run
    python scripts/cleanup_evp_intermediates.py
    python scripts/cleanup_evp_intermediates.py --archive ./evp_archive
    python scripts/cleanup_evp_intermediates.py --update-yaml
"""

import argparse
import glob
import os
import shutil
import sys

import yaml


def find_intermediate_files(base_path: str):
    """Find segment and blended WAV files in chain_artifacts."""
    # Check both audio/ and wav/ subdirs; filenames are {uuid}_z_segment_{n}.wav
    segment_files = []
    blended_files = []
    for subdir in ("audio", "wav"):
        segment_files.extend(
            glob.glob(os.path.join(base_path, f"*/{subdir}/*_segment_*.wav"))
        )
        blended_files.extend(
            glob.glob(os.path.join(base_path, f"*/{subdir}/blended*.wav"))
        )
    segments = sorted(segment_files)
    blended = sorted(blended_files)
    return segments, blended


def get_file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def format_bytes(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def update_evp_yaml_files(base_path: str, dry_run: bool = False):
    """Remove stale audio_segments and noise_blended_audio from EVP YAML files."""
    yaml_pattern = os.path.join(base_path, "*/yml/evp_*.yml")
    yaml_files = sorted(glob.glob(yaml_pattern))
    updated = 0
    for yf in yaml_files:
        with open(yf, "r") as f:
            data = yaml.safe_load(f)
        if data is None:
            continue
        changed = False
        for key in ("audio_segments", "noise_blended_audio"):
            if key in data:
                changed = True
                del data[key]
        if changed:
            if dry_run:
                print(f"  Would update: {yf}")
            else:
                with open(yf, "w") as f:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
                print(f"  Updated: {yf}")
            updated += 1
    return updated


def main():
    parser = argparse.ArgumentParser(
        description="Clean up legacy EVP intermediate audio files"
    )
    parser.add_argument(
        "--base-path",
        default="chain_artifacts",
        help="Base path for chain artifacts (default: chain_artifacts)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be deleted without making changes",
    )
    parser.add_argument(
        "--archive",
        metavar="PATH",
        help="Move files to archive directory instead of deleting",
    )
    parser.add_argument(
        "--update-yaml",
        action="store_true",
        help="Also remove stale fields from EVP YAML files",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.base_path):
        print(f"Base path not found: {args.base_path}")
        sys.exit(1)

    segments, blended = find_intermediate_files(args.base_path)
    all_files = segments + blended
    total_size = sum(get_file_size(f) for f in all_files)

    print(f"Found {len(segments)} segment files and {len(blended)} blended files")
    print(f"Total space: {format_bytes(total_size)}")

    if not all_files and not args.update_yaml:
        print("Nothing to clean up.")
        return

    if args.dry_run:
        print("\n[DRY RUN] Would remove:")
        for f in all_files:
            print(f"  {f} ({format_bytes(get_file_size(f))})")
    elif args.archive:
        os.makedirs(args.archive, exist_ok=True)
        for f in all_files:
            dest = os.path.join(args.archive, os.path.basename(f))
            shutil.move(f, dest)
            print(f"  Archived: {f} -> {dest}")
        print(f"\nArchived {len(all_files)} files to {args.archive}")
    else:
        for f in all_files:
            os.remove(f)
            print(f"  Deleted: {f}")
        print(f"\nDeleted {len(all_files)} files, freed {format_bytes(total_size)}")

    if args.update_yaml:
        print("\nUpdating EVP YAML files...")
        updated = update_evp_yaml_files(args.base_path, dry_run=args.dry_run)
        print(f"{'Would update' if args.dry_run else 'Updated'} {updated} YAML files")


if __name__ == "__main__":
    main()
