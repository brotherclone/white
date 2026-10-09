"""Archive Logic song folders that aren't placed on an LP side.

Moves `$LOGIC_OUTPUT_DIR/<thread>/<folder>/` to `$LOGIC_ARCHIVE_DIR/<thread>/<folder>/`
so the fast production drive only holds songs still in play. A song is *placed*
when it appears in `sides.yml`; `archive_keep.yml` (album root) protects extra
songs or whole threads, e.g. interstitial sources. Everything else is a
candidate.

Each move is copy → verify (size + SHA-256 per file) → rewrite stored absolute
paths → record in `archive_manifest.yml` → delete source. A failure before the
delete leaves the source untouched. Dry-run by default; pass `--execute`.

`logic_handoff.resolve_song_dir()` falls back to the archive, so archived songs
keep working with the board, /composition and re-handoff.

Usage:
    python -m white_composition.logic_archive                      # dry run
    python -m white_composition.logic_archive --execute --only <song_id>
    python -m white_composition.logic_archive --execute --limit 10
    python -m white_composition.logic_archive --restore <song_id> --execute
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

from white_composition.lp_sides import load_sides
from white_core.enums.archive_classification import ArchiveClassification
from white_core.enums.archive_status import ArchiveStatus
from white_core.music.core.logic_archive import (
    ArchiveKeepList,
    ArchiveManifest,
    ArchiveManifestEntry,
)

ARCHIVE_MANIFEST_FILENAME = "archive_manifest.yml"
ARCHIVE_KEEP_FILENAME = "archive_keep.yml"
PARTIAL_SUFFIX = ".partial"
IGNORABLE_THREAD_FILES = {".DS_Store"}
COMPOSITION_FILENAME = "composition.yml"
SONG_CONTEXT_FILENAME = "song_context.yml"

_SLUG_SUFFIX_RE = re.compile(r"\(([^()]+)\)$")
_HASH_CHUNK = 1024 * 1024


class LogicArchiveOfflineError(RuntimeError):
    """A song is recorded as archived but the archive volume isn't reachable."""


class ArchiveVerificationError(RuntimeError):
    """The copied tree doesn't match the source."""


@dataclass
class SongFolder:
    thread: str
    folder_name: str
    path: Path
    song_id: str | None
    production_dir: Path | None
    classification: ArchiveClassification
    size_bytes: int


# ---------------------------------------------------------------------------
# Config / persistence
# ---------------------------------------------------------------------------


def logic_archive_dir() -> Path | None:
    """`LOGIC_ARCHIVE_DIR` from the environment, or None if unset."""
    val = os.environ.get("LOGIC_ARCHIVE_DIR", "")
    return Path(val) if val else None


def _dump_yaml(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        yaml.dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=float("inf"),
        )


def load_manifest(album_dir: Path) -> ArchiveManifest:
    path = Path(album_dir) / ARCHIVE_MANIFEST_FILENAME
    if not path.exists():
        return ArchiveManifest()
    with open(path) as f:
        return ArchiveManifest.model_validate(yaml.safe_load(f) or {})


def save_manifest(album_dir: Path, manifest: ArchiveManifest) -> Path:
    path = Path(album_dir) / ARCHIVE_MANIFEST_FILENAME
    _dump_yaml(manifest.model_dump(mode="json"), path)
    return path


def load_keep_list(album_dir: Path) -> ArchiveKeepList:
    path = Path(album_dir) / ARCHIVE_KEEP_FILENAME
    if not path.exists():
        return ArchiveKeepList()
    with open(path) as f:
        return ArchiveKeepList.model_validate(yaml.safe_load(f) or {})


def split_song_id(song_id: str) -> tuple[str, str]:
    """`<thread>__<production_slug>` → (thread, slug). Thread slugs never contain `__`."""
    thread, sep, slug = song_id.partition("__")
    if not sep:
        raise ValueError(f"Malformed song_id '{song_id}'")
    return thread, slug


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _tree_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            p = Path(root) / name
            if not p.is_symlink():
                total += p.stat().st_size
    return total


def classify(logic_root: Path, album_dir: Path) -> list[SongFolder]:
    """Classify every `<thread>/<folder>` under *logic_root*."""
    logic_root = Path(logic_root)
    album_dir = Path(album_dir)

    placed_ids = {
        s.song_id for side in load_sides(album_dir).sides.values() for s in side.songs
    }
    # Matching on slug as well as full id is deliberately conservative: a song
    # whose Logic thread folder differs from its shrink_wrapped thread is still
    # treated as placed.
    placed_slugs = {split_song_id(i)[1] for i in placed_ids if "__" in i}
    keep = load_keep_list(album_dir)
    keep_slugs = {split_song_id(i)[1] for i in keep.songs if "__" in i}

    folders: list[SongFolder] = []
    if not logic_root.is_dir():
        return folders
    for thread_dir in sorted(p for p in logic_root.iterdir() if p.is_dir()):
        for song_path in sorted(p for p in thread_dir.iterdir() if p.is_dir()):
            if song_path.name.endswith(PARTIAL_SUFFIX):
                continue
            thread = thread_dir.name
            match = _SLUG_SUFFIX_RE.search(song_path.name)
            slug = match.group(1) if match else None
            prod_dir = album_dir / thread / "production" / slug if slug else None
            song_id = f"{thread}__{slug}" if slug else None

            if song_id in placed_ids or (slug and slug in placed_slugs):
                classification = ArchiveClassification.PLACED
            elif (
                song_id in keep.songs
                or thread in keep.threads
                or (slug and slug in keep_slugs)
            ):
                classification = ArchiveClassification.KEPT
            elif prod_dir is None or not prod_dir.is_dir():
                classification = ArchiveClassification.UNMATCHED
                prod_dir = None
            else:
                classification = ArchiveClassification.CANDIDATE

            folders.append(
                SongFolder(
                    thread=thread,
                    folder_name=song_path.name,
                    path=song_path,
                    song_id=song_id,
                    production_dir=prod_dir,
                    classification=classification,
                    size_bytes=_tree_size(song_path),
                )
            )
    return folders


# ---------------------------------------------------------------------------
# Verified move
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _inventory(root: Path) -> dict[str, tuple[tuple, Path]]:
    """NFC relative path → (("dir",) | ("link", target) | ("file", size), real path).

    Keys are NFC-normalised because HFS+ volumes store names decomposed (NFD)
    while APFS preserves whatever was written — `Séance` would otherwise look
    like a missing file plus an unexpected one.
    """
    items: dict[str, tuple[tuple, Path]] = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(dirpath)
        for name in dirnames + filenames:
            p = base / name
            rel = unicodedata.normalize("NFC", str(p.relative_to(root)))
            if p.is_symlink():
                entry = ("link", unicodedata.normalize("NFC", os.readlink(p)))
            elif p.is_dir():
                entry = ("dir",)
            else:
                entry = ("file", p.stat().st_size)
            items[rel] = (entry, p)
    return items


def verify_copy(src: Path, dst: Path) -> tuple[int, int]:
    """Raise ArchiveVerificationError unless *dst* mirrors *src*. Returns (file_count, bytes)."""
    src_inv = _inventory(src)
    dst_inv = _inventory(dst)
    if src_inv.keys() != dst_inv.keys():
        missing = sorted(src_inv.keys() - dst_inv.keys())[:5]
        extra = sorted(dst_inv.keys() - src_inv.keys())[:5]
        raise ArchiveVerificationError(
            f"Tree mismatch — missing {missing}, unexpected {extra}"
        )
    file_count = 0
    total = 0
    for rel, (entry, src_path) in src_inv.items():
        dst_entry, dst_path = dst_inv[rel]
        if entry != dst_entry:
            raise ArchiveVerificationError(f"Metadata mismatch: {rel}")
        if entry[0] == "file":
            if _sha256(src_path) != _sha256(dst_path):
                raise ArchiveVerificationError(f"Checksum mismatch: {rel}")
            file_count += 1
            total += entry[1]
        elif entry[0] == "link":
            file_count += 1
    return file_count, total


def _rewrite_prefix(value, old: str, new: str):
    if isinstance(value, str):
        if value == old or value.startswith(old + "/"):
            return new + value[len(old) :]
        return value
    if isinstance(value, dict):
        return {k: _rewrite_prefix(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [_rewrite_prefix(v, old, new) for v in value]
    return value


def rewrite_yaml_paths(path: Path, old: Path, new: Path) -> bool:
    """Rewrite string values under *old* to *new* in a YAML file. Returns True if changed."""
    if not path.exists():
        return False
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return False
    updated = _rewrite_prefix(data, str(old), str(new))
    if updated == data:
        return False
    _dump_yaml(updated, path)
    return True


def set_logic_project_path(song_dir: Path, value: Path) -> bool:
    """Point composition.yml's `logic_project_path` at *value*. Returns True if changed.

    Set outright rather than prefix-rewritten: composition.yml files from older
    handoffs recorded the folder name without its `(<slug>)` suffix, so their
    stored path was already stale and wouldn't match the old prefix.
    """
    path = Path(song_dir) / COMPOSITION_FILENAME
    if not path.exists():
        return False
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or data.get("logic_project_path") == str(value):
        return False
    data["logic_project_path"] = str(value)
    _dump_yaml(data, path)
    return True


def _cleanup_thread_dir(thread_dir: Path) -> None:
    """Remove *thread_dir* if it holds nothing but ignorable Finder files."""
    if not thread_dir.is_dir():
        return
    entries = list(thread_dir.iterdir())
    if any(e.name not in IGNORABLE_THREAD_FILES for e in entries):
        return
    for e in entries:
        e.unlink()
    thread_dir.rmdir()


def move_song_folder(
    src: Path, dst: Path, production_dir: Path | None
) -> tuple[int, int]:
    """Copy → verify → rewrite paths → rename → delete source. Returns (file_count, bytes).

    On any failure before the source is deleted, the destination copy is removed,
    any song_context.yml rewrite is reverted, and the source is left as-is.
    """
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    partial = dst.with_name(dst.name + PARTIAL_SUFFIX)
    if partial.exists():
        shutil.rmtree(partial)  # stale leftover from an interrupted run
    partial.parent.mkdir(parents=True, exist_ok=True)

    ctx_path = production_dir / SONG_CONTEXT_FILENAME if production_dir else None
    ctx_backup = ctx_path.read_text() if ctx_path and ctx_path.exists() else None
    try:
        shutil.copytree(src, partial, symlinks=True)
        file_count, total = verify_copy(src, partial)
        set_logic_project_path(partial, dst)
        partial.rename(dst)
        if ctx_path is not None:
            rewrite_yaml_paths(ctx_path, src, dst)
    except BaseException:
        for leftover in (partial, dst):
            if leftover.exists():
                shutil.rmtree(leftover)
        if ctx_backup is not None:
            ctx_path.write_text(ctx_backup)
        _cleanup_thread_dir(dst.parent)
        raise

    shutil.rmtree(src)
    _cleanup_thread_dir(src.parent)
    return file_count, total


def _record(
    album_dir: Path,
    song_id: str,
    source: Path,
    dest: Path,
    file_count: int,
    total: int,
    status: ArchiveStatus,
) -> None:
    manifest = load_manifest(album_dir)
    manifest.songs[song_id] = ArchiveManifestEntry(
        source_path=str(source),
        dest_path=str(dest),
        file_count=file_count,
        total_bytes=total,
        moved_at=datetime.now(timezone.utc),
        status=status,
    )
    save_manifest(album_dir, manifest)


def archive_song(folder: SongFolder, archive_root: Path, album_dir: Path) -> Path:
    dst = Path(archive_root) / folder.thread / folder.folder_name
    file_count, total = move_song_folder(folder.path, dst, folder.production_dir)
    _record(
        album_dir,
        folder.song_id,
        folder.path,
        dst,
        file_count,
        total,
        ArchiveStatus.ARCHIVED,
    )
    return dst


def restore_song(song_id: str, album_dir: Path) -> Path:
    manifest = load_manifest(album_dir)
    entry = manifest.songs.get(song_id)
    if entry is None or entry.status != ArchiveStatus.ARCHIVED:
        raise KeyError(f"'{song_id}' is not recorded as archived")
    src = Path(entry.dest_path)
    dst = Path(entry.source_path)
    if not src.exists():
        raise LogicArchiveOfflineError(
            f"Archived folder not reachable: {src} — is the archive volume mounted?"
        )
    thread, slug = split_song_id(song_id)
    prod_dir = Path(album_dir) / thread / "production" / slug
    file_count, total = move_song_folder(
        src, dst, prod_dir if prod_dir.is_dir() else None
    )
    _record(album_dir, song_id, src, dst, file_count, total, ArchiveStatus.RESTORED)
    return dst


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _gb(n: int) -> str:
    return f"{n / 1024**3:.2f} GB"


def _print_report(folders: list[SongFolder], archive_root: Path | None) -> None:
    for cls in ArchiveClassification:
        group = [f for f in folders if f.classification == cls]
        size = sum(f.size_bytes for f in group)
        print(f"\n{cls.value.upper()} ({len(group)}, {_gb(size)})")
        for f in group:
            print(f"  {_gb(f.size_bytes):>10}  {f.thread}/{f.folder_name}")
    print(f"\nArchive destination: {archive_root or '(LOGIC_ARCHIVE_DIR not set)'}")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Archive unplaced Logic song folders (dry run by default)."
    )
    parser.add_argument(
        "--album-dir",
        type=Path,
        default=os.environ.get("SHRINKWRAP_OUTPUT_DIR"),
        help="Album root holding sides.yml (default: $SHRINKWRAP_OUTPUT_DIR)",
    )
    parser.add_argument("--execute", action="store_true", help="Actually move folders")
    parser.add_argument("--only", help="Restrict to a single song_id")
    parser.add_argument("--limit", type=int, help="Move at most N songs")
    parser.add_argument(
        "--include-unmatched",
        action="store_true",
        help="Also archive folders with no matching production dir",
    )
    parser.add_argument("--restore", metavar="SONG_ID", help="Move a song back")
    args = parser.parse_args(argv)

    if not args.album_dir:
        parser.error("--album-dir or SHRINKWRAP_OUTPUT_DIR is required")
    album_dir = Path(args.album_dir)
    archive_root = logic_archive_dir()

    if args.restore:
        if not args.execute:
            entry = load_manifest(album_dir).songs.get(args.restore)
            if entry is None:
                print(f"'{args.restore}' is not in the archive manifest.")
                return 1
            print(f"Would restore {entry.dest_path} → {entry.source_path}")
            return 0
        dst = restore_song(args.restore, album_dir)
        print(f"Restored → {dst}")
        return 0

    logic_root = os.environ.get("LOGIC_OUTPUT_DIR", "")
    if not logic_root:
        raise EnvironmentError("LOGIC_OUTPUT_DIR is not set — add it to .env")
    folders = classify(Path(logic_root), album_dir)
    if args.only:
        folders = [f for f in folders if f.song_id == args.only]
        if not folders:
            print(f"No Logic folder found for '{args.only}'.")
            return 1

    movable = {ArchiveClassification.CANDIDATE}
    if args.include_unmatched:
        movable.add(ArchiveClassification.UNMATCHED)
    to_move = [f for f in folders if f.classification in movable]
    if args.limit is not None:
        to_move = to_move[: args.limit]

    if not args.execute:
        _print_report(folders, archive_root)
        print(
            f"\nDry run — {len(to_move)} folder(s), {_gb(sum(f.size_bytes for f in to_move))} "
            "would be archived. Re-run with --execute."
        )
        return 0

    if archive_root is None:
        raise EnvironmentError("LOGIC_ARCHIVE_DIR is not set — add it to .env")
    if not archive_root.is_dir():
        raise EnvironmentError(
            f"LOGIC_ARCHIVE_DIR does not exist: {archive_root} — is the volume mounted?"
        )

    moved, failed, conflicts = 0, [], []
    for i, folder in enumerate(to_move, 1):
        label = f"[{i}/{len(to_move)}] {folder.thread}/{folder.folder_name}"
        if folder.song_id is None:
            # Unmatched with no slug suffix — key the manifest on the folder path.
            folder.song_id = f"{folder.thread}__{folder.folder_name}"
        try:
            dst = archive_song(folder, archive_root, album_dir)
        except FileExistsError:
            conflicts.append(label)
            print(f"{label}  CONFLICT — destination exists, skipped")
            continue
        except Exception as exc:
            failed.append(label)
            print(f"{label}  FAILED — {exc} (source untouched)")
            continue
        moved += 1
        print(f"{label}  archived ({_gb(folder.size_bytes)}) → {dst}")

    print(f"\nArchived {moved}, conflicts {len(conflicts)}, failed {len(failed)}.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
