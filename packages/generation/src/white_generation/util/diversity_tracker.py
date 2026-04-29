"""
Album-level template diversity tracker.

Prevents melodic/bass sameness across songs by penalising templates that have
been used frequently and rewarding ones that haven't appeared yet.

The registry is a simple JSON file (used_templates.json) placed in the
shrink_wrapped/ album root directory.  Both the melody and bass pipelines read
it during candidate scoring and promote_part writes back when a candidate is
approved.

Score multipliers
-----------------
- Template used in 0 songs   → 1.1×  (novelty bonus)
- Template used in 1–2 songs → 1.0×  (neutral)
- Template used in 3+ songs  → 0.75× (repetition penalty)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

REGISTRY_FILENAME = "used_templates.json"
PENALTY_THRESHOLD = 3  # uses at or above this → apply penalty
PENALTY_FACTOR = 0.75
BONUS_FACTOR = 1.1


# ---------------------------------------------------------------------------
# Album-dir discovery
# ---------------------------------------------------------------------------


def find_album_dir(path: Path) -> Optional[Path]:
    """Walk up from *path* to find the shrink_wrapped/ directory.

    Returns the shrink_wrapped/ Path if found, or None if not in a
    shrink_wrapped tree.
    """
    p = path.resolve()
    while p != p.parent:
        if p.name == "shrink_wrapped":
            return p
        p = p.parent
    return None


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------


def load_registry(album_dir: Path) -> dict[str, int]:
    """Return {template_name: song_count} from album_dir/used_templates.json."""
    registry_path = album_dir / REGISTRY_FILENAME
    if registry_path.exists():
        try:
            return json.loads(registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_registry(album_dir: Path, registry: dict[str, int]) -> None:
    """Write the registry back to album_dir/used_templates.json."""
    registry_path = album_dir / REGISTRY_FILENAME
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------


def diversity_factor(template_name: str, registry: dict[str, int]) -> float:
    """Return the score multiplier for a template given the current registry.

    0 prior uses   → BONUS_FACTOR  (1.1)
    1–2 prior uses → 1.0
    3+ prior uses  → PENALTY_FACTOR (0.75)
    """
    count = registry.get(template_name, 0)
    if count == 0:
        return BONUS_FACTOR
    if count >= PENALTY_THRESHOLD:
        return PENALTY_FACTOR
    return 1.0


# ---------------------------------------------------------------------------
# Record use
# ---------------------------------------------------------------------------


def record_use(template_name: str, registry: dict[str, int]) -> dict[str, int]:
    """Increment the song-count for *template_name* in *registry* and return it."""
    registry[template_name] = registry.get(template_name, 0) + 1
    return registry
