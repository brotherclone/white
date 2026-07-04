"""LP-side sequencing — bucket finished mixes into 4 sides (A-D) against a soft duration limit.

White is the only double album in the catalog. This module tracks which songs are
assigned to which of the 4 physical-LP sides (A-D), caches each assignment's mix
duration (read via `soundfile`), and computes per-side running totals against a
soft (non-enforced) `side_limit_seconds`, default 1200 (20 minutes).

`sides.yml` lives at the album root, alongside `index.yml` and
`negative_constraints.yml`.
"""

from pathlib import Path
from typing import Optional

import soundfile as sf
import yaml
from pydantic import BaseModel

SIDES_FILENAME = "sides.yml"
SIDE_NAMES = ["A", "B", "C", "D"]
DEFAULT_SIDE_LIMIT_SECONDS = 1200.0


class SideSong(BaseModel):
    song_id: str
    duration_seconds: float


class Side(BaseModel):
    songs: list[SideSong] = []


class SidesDocument(BaseModel):
    side_limit_seconds: float = DEFAULT_SIDE_LIMIT_SECONDS
    sides: dict[str, Side]

    @classmethod
    def empty(cls) -> "SidesDocument":
        return cls(sides={name: Side() for name in SIDE_NAMES})


def mix_duration_seconds(path: Path) -> Optional[float]:
    """Return mix audio duration in seconds, or None if missing/unreadable."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        info = sf.info(str(path))
        return round(info.frames / info.samplerate, 3)
    except Exception:
        return None


def load_sides(album_dir: Path) -> SidesDocument:
    """Load sides.yml from the album root, or an empty 4-side document if absent."""
    path = Path(album_dir) / SIDES_FILENAME
    if not path.exists():
        return SidesDocument.empty()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    raw_sides = data.get("sides") or {}
    sides = {
        name: Side(
            songs=[SideSong(**s) for s in (raw_sides.get(name) or {}).get("songs", [])]
        )
        for name in SIDE_NAMES
    }
    return SidesDocument(
        side_limit_seconds=data.get("side_limit_seconds", DEFAULT_SIDE_LIMIT_SECONDS),
        sides=sides,
    )


def save_sides(album_dir: Path, doc: SidesDocument) -> Path:
    """Write sides.yml to the album root."""
    path = Path(album_dir) / SIDES_FILENAME
    data = {
        "side_limit_seconds": doc.side_limit_seconds,
        "sides": {
            name: {
                "songs": [
                    {"song_id": s.song_id, "duration_seconds": s.duration_seconds}
                    for s in side.songs
                ]
            }
            for name, side in doc.sides.items()
        },
    }
    with open(path, "w") as f:
        yaml.dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=float("inf"),
        )
    return path


def _find_song_side(doc: SidesDocument, song_id: str) -> Optional[str]:
    for name, side in doc.sides.items():
        if any(s.song_id == song_id for s in side.songs):
            return name
    return None


def assign_song(
    doc: SidesDocument,
    song_id: str,
    side: str,
    position: int,
    duration_seconds: float,
) -> SidesDocument:
    """Insert song_id into `side` at `position`, removing it from any other side first."""
    if side not in doc.sides:
        raise ValueError(f"Unknown side '{side}'")
    remove_song(doc, song_id)
    songs = doc.sides[side].songs
    position = max(0, min(position, len(songs)))
    songs.insert(position, SideSong(song_id=song_id, duration_seconds=duration_seconds))
    return doc


def move_song(
    doc: SidesDocument, song_id: str, to_side: str, to_position: int
) -> SidesDocument:
    """Move an already-assigned song to a (possibly different) side/position.

    Preserves the song's previously cached duration.
    """
    current_side = _find_song_side(doc, song_id)
    if current_side is None:
        raise ValueError(f"Song '{song_id}' is not assigned to any side")
    existing = next(s for s in doc.sides[current_side].songs if s.song_id == song_id)
    return assign_song(doc, song_id, to_side, to_position, existing.duration_seconds)


def remove_song(doc: SidesDocument, song_id: str) -> SidesDocument:
    """Remove song_id from whichever side it's currently on, if any."""
    for side in doc.sides.values():
        side.songs = [s for s in side.songs if s.song_id != song_id]
    return doc


def side_totals(doc: SidesDocument) -> dict[str, dict]:
    """Return per-side {total_seconds, over_limit} summary."""
    totals = {}
    for name, side in doc.sides.items():
        total = round(sum(s.duration_seconds for s in side.songs), 3)
        totals[name] = {
            "total_seconds": total,
            "over_limit": total > doc.side_limit_seconds,
        }
    return totals
