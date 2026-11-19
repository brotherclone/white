"""Listening-playlist sync — bucket songs with finished mixes into Rejects /
Review / White Album WiP folders for Apple Music import.

Classification is derived fresh from `scan_songs()` output and `sides.yml` on every
sync; no bucket membership is persisted. Only the output directory setting persists,
in `playlist_config.yml` at the album root (sibling to `sides.yml`).
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

import yaml
from mutagen.id3 import ID3, TALB, TRCK, ID3NoHeaderError
from pydantic import BaseModel

from white_composition.lp_sides import SIDE_NAMES, SidesDocument, mix_duration_seconds

PLAYLIST_CONFIG_FILENAME = "playlist_config.yml"
DEFAULT_OUTPUT_DIR = str(
    Path.home()
    / "Documents"
    / "Music Production"
    / "Earthly Frames"
    / "White"
    / "Listening"
)
DEFAULT_MATERIAL_TARGET_SECONDS = 9 * 3600.0
DEFAULT_WIP_ALBUM_TITLE = "White Album WiP"

REJECTS_DIRNAME = "Rejects"
REVIEW_DIRNAME = "Review"
WIP_DIRNAME = "White Album WiP"

_LIFECYCLE_REJECT_STATUSES = {"scrapped", "abandoned"}
_AUDIO_EXTENSIONS = {".mp3", ".wav", ".aiff", ".aif", ".m4a"}
_DEFAULT_EXTENSION = ".mp3"

logger = logging.getLogger(__name__)


class PlaylistConfig(BaseModel):
    output_dir: str = DEFAULT_OUTPUT_DIR
    wip_album_title: str = DEFAULT_WIP_ALBUM_TITLE


class WipTag(BaseModel):
    """ID3 tags stamped on a White Album WiP mp3 copy. `track` is None for
    unsequenced songs (their copy gets no track number)."""

    album: str
    track: int | None = None
    total: int | None = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_playlist_config(album_dir: Path) -> PlaylistConfig:
    """Load playlist_config.yml from the album root, materializing defaults if absent."""
    path = Path(album_dir) / PLAYLIST_CONFIG_FILENAME
    if not path.exists():
        config = PlaylistConfig()
        save_playlist_config(album_dir, config)
        return config
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return PlaylistConfig(**data)


def save_playlist_config(album_dir: Path, config: PlaylistConfig) -> Path:
    """Write playlist_config.yml to the album root."""
    path = Path(album_dir) / PLAYLIST_CONFIG_FILENAME
    with open(path, "w") as f:
        yaml.dump(
            config.model_dump(),
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=float("inf"),
        )
    return path


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _is_rejected(song: dict) -> bool:
    return song.get("lifecycle_status") in _LIFECYCLE_REJECT_STATUSES


def classify_songs(
    songs: list[dict], sides_doc: SidesDocument
) -> dict[str, list[dict]]:
    """Classify songs (from `scan_songs()`) into rejects/review/wip buckets.

    Only songs with `has_mix: true` are considered; each song lands in at most one
    bucket. WiP songs are ordered by Side A-D + in-side position, each annotated with
    a `_side` key; a placed song absent from `sides.yml` is appended at the end with
    `_side: None` (unsequenced) rather than dropped.
    """
    rejects: list[dict] = []
    review: list[dict] = []
    wip_by_id: dict[str, dict] = {}

    for song in songs:
        if not song.get("has_mix"):
            continue
        if song.get("lp_consideration") == "placed":
            wip_by_id[song["id"]] = song
        elif _is_rejected(song):
            rejects.append(song)
        else:
            review.append(song)

    wip_sequenced: list[dict] = []
    seen_ids: set[str] = set()
    for side_name in SIDE_NAMES:
        side = sides_doc.sides.get(side_name)
        if side is None:
            continue
        for side_song in side.songs:
            song = wip_by_id.get(side_song.song_id)
            if song is not None and side_song.song_id not in seen_ids:
                wip_sequenced.append({**song, "_side": side_name})
                seen_ids.add(side_song.song_id)

    wip_unsequenced = [
        {**song, "_side": None}
        for song_id, song in wip_by_id.items()
        if song_id not in seen_ids
    ]

    return {
        "rejects": rejects,
        "review": review,
        "wip": wip_sequenced + wip_unsequenced,
    }


# ---------------------------------------------------------------------------
# Filesystem rebuild
# ---------------------------------------------------------------------------


def _sanitize_filename(name: str) -> str:
    """Strip characters invalid in filenames, leaving a reasonable stem."""
    name = re.sub(r'[\/:*?"<>|\x00-\x1f]', "_", name).strip()
    return name or "untitled"


def read_mix_path(production_path: str) -> Path | None:
    """Return the song's mix file path from song_context.yml, if set and present."""
    ctx_path = Path(production_path) / "song_context.yml"
    if not ctx_path.exists():
        return None
    try:
        with open(ctx_path) as f:
            ctx = yaml.safe_load(f) or {}
    except Exception:
        return None
    mix = ctx.get("mix_file")
    if not mix:
        return None
    mix_path = Path(mix)
    return mix_path if mix_path.exists() else None


def _mix_extension(song: dict) -> str:
    mix_path = read_mix_path(song["production_path"])
    return mix_path.suffix if mix_path else _DEFAULT_EXTENSION


def _dedupe_filenames(entries: list[tuple[dict, str, str]]) -> dict[str, str]:
    """entries: (song, stem, ext) triples. Returns {song_id: filename}, suffixing
    stems that collide with the song's production_slug."""
    stem_counts: dict[str, int] = {}
    for _song, stem, _ext in entries:
        stem_counts[stem] = stem_counts.get(stem, 0) + 1
    result: dict[str, str] = {}
    for song, stem, ext in entries:
        if stem_counts[stem] > 1:
            suffix = song.get("production_slug") or song["id"]
            result[song["id"]] = f"{stem}_{suffix}{ext}"
        else:
            result[song["id"]] = f"{stem}{ext}"
    return result


def _plain_bucket_filenames(songs: list[dict], prefix: str = "") -> dict[str, str]:
    entries = [
        (song, f"{prefix}{_sanitize_filename(song['title'])}", _mix_extension(song))
        for song in songs
    ]
    return _dedupe_filenames(entries)


def _wip_filenames(wip_songs: list[dict]) -> dict[str, str]:
    """Sequenced entries are already unique via their index prefix. Unsequenced
    entries all share the "99_unsequenced_" prefix, so — unlike the sequenced
    ones — they need the same collision-suffix handling as the Rejects/Review
    buckets to avoid two same-titled unsequenced songs overwriting each other.
    """
    names: dict[str, str] = {}
    unsequenced_entries: list[tuple[dict, str, str]] = []
    for i, song in enumerate(wip_songs):
        stem = _sanitize_filename(song["title"])
        ext = _mix_extension(song)
        if song.get("_side") is not None:
            names[song["id"]] = f"{i + 1:02d}_{song['_side']}_{stem}{ext}"
        else:
            unsequenced_entries.append((song, f"99_unsequenced_{stem}", ext))
    names.update(_dedupe_filenames(unsequenced_entries))
    return names


def _wip_tags(wip_songs: list[dict], album: str) -> dict[str, WipTag]:
    """Album + track tags for each WiP song. Track numbers follow the same global
    1-based sequence as the `{seq:02d}_` filename prefix; unsequenced songs get
    the album tag only."""
    total = sum(1 for s in wip_songs if s.get("_side") is not None)
    tags: dict[str, WipTag] = {}
    for i, song in enumerate(wip_songs):
        if song.get("_side") is not None:
            tags[song["id"]] = WipTag(album=album, track=i + 1, total=total)
        else:
            tags[song["id"]] = WipTag(album=album)
    return tags


def _tag_wip_copy(path: Path, tag: WipTag) -> None:
    """Write album/track ID3 tags to a copied WiP mp3. Never call on a source
    mix — only on the copy in the WiP folder."""
    try:
        id3 = ID3(str(path))
    except ID3NoHeaderError:
        id3 = ID3()
    id3.setall("TALB", [TALB(encoding=3, text=tag.album)])
    if tag.track is not None:
        id3.setall("TRCK", [TRCK(encoding=3, text=f"{tag.track}/{tag.total}")])
    else:
        id3.delall("TRCK")
    id3.save(str(path))


def _validate_output_dir(output_dir: str) -> Path:
    if not output_dir or not output_dir.strip():
        raise ValueError("Playlist output_dir is not configured")
    resolved = Path(output_dir).expanduser().resolve()
    if str(resolved) in ("/", str(Path.home())):
        raise ValueError(f"Refusing to sync to unsafe output_dir: {resolved}")
    return resolved


def _rebuild_folder(
    folder: Path,
    target_names: dict[str, str],
    songs_by_id: dict[str, dict],
    tags: dict[str, WipTag] | None = None,
) -> int:
    """Empty `folder` of prior audio files, then copy in `target_names`
    {song_id: filename} fresh. Returns count synced.

    Rebuilding from an empty folder (rather than diffing against what's already
    there) avoids stale duplicates left behind when a song moves between sides
    or buckets and its filename changes.

    When `tags` is given, each copied `.mp3` whose song has an entry is tagged
    in place; a tagging failure is logged and leaves the untagged copy.
    """
    folder.mkdir(parents=True, exist_ok=True)

    for existing in folder.iterdir():
        if existing.is_file() and existing.suffix.lower() in _AUDIO_EXTENSIONS:
            existing.unlink()

    count = 0
    for song_id, filename in target_names.items():
        song = songs_by_id[song_id]
        mix_path = read_mix_path(song["production_path"])
        if mix_path is None:
            continue
        dest = folder / filename
        shutil.copy2(mix_path, dest)
        count += 1
        tag = (tags or {}).get(song_id)
        if tag is not None and dest.suffix.lower() == ".mp3":
            try:
                _tag_wip_copy(dest, tag)
            except Exception as e:
                logger.warning("Could not tag WiP copy %s: %s", dest, e)
    return count


def sync_playlists(
    songs: list[dict],
    sides_doc: SidesDocument,
    output_dir: str,
    wip_album_title: str = DEFAULT_WIP_ALBUM_TITLE,
) -> dict[str, int]:
    """Classify songs and rebuild the 3 playlist folders under output_dir.

    White Album WiP mp3 copies are tagged with `wip_album_title` and a track
    number from the sides.yml sequence; Rejects/Review copies are untouched.

    Returns {"rejects": n, "review": n, "wip": n} — the number of files present in
    each subfolder after the rebuild.
    """
    base = _validate_output_dir(output_dir)
    buckets = classify_songs(songs, sides_doc)

    rejects_names = _plain_bucket_filenames(buckets["rejects"], prefix="REJECT_")
    review_names = _plain_bucket_filenames(buckets["review"])
    wip_names = _wip_filenames(buckets["wip"])

    songs_by_id = {s["id"]: s for s in songs}
    for s in buckets["wip"]:
        songs_by_id.setdefault(s["id"], s)

    return {
        "rejects": _rebuild_folder(base / REJECTS_DIRNAME, rejects_names, songs_by_id),
        "review": _rebuild_folder(base / REVIEW_DIRNAME, review_names, songs_by_id),
        "wip": _rebuild_folder(
            base / WIP_DIRNAME,
            wip_names,
            songs_by_id,
            tags=_wip_tags(buckets["wip"], wip_album_title),
        ),
    }


# ---------------------------------------------------------------------------
# Material produced
# ---------------------------------------------------------------------------


def material_produced_seconds(output_dir: str) -> tuple[float, int]:
    """Sum durations of mp3s directly under `output_dir` (Rejects/Review/WiP
    subfolders excluded — this counts only tracks the user has placed loose in
    the Listening folder itself). Returns (total_seconds, file_count)."""
    base = _validate_output_dir(output_dir)
    if not base.is_dir():
        return 0.0, 0
    total = 0.0
    count = 0
    for entry in base.iterdir():
        if not entry.is_file() or entry.suffix.lower() != ".mp3":
            continue
        duration = mix_duration_seconds(entry)
        if duration is None:
            continue
        total += duration
        count += 1
    return round(total, 3), count
