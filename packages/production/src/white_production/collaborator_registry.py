from __future__ import annotations

import os
from pathlib import Path

import yaml

from white_core.music.core.collaborator import Collaborator

_SRC_ROOT = Path(__file__).parent


def _registry_dir() -> Path:
    root = os.environ.get("WHITE_PROJECT_ROOT")
    if root:
        base = Path(root)
    else:
        # 4 levels up from src/white_production/ → project root
        base = _SRC_ROOT.parent.parent.parent.parent
    return (
        base
        / "packages"
        / "core"
        / "src"
        / "white_core"
        / "music"
        / "core"
        / "collaborators"
    )


def load_collaborator(
    collaborator_id: str, registry_dir: Path | None = None
) -> Collaborator:
    d = registry_dir or _registry_dir()
    path = d / f"{collaborator_id}.yml"
    if not path.exists():
        raise FileNotFoundError(f"No collaborator found with id '{collaborator_id}'")
    raw = yaml.safe_load(path.read_text()) or {}
    return Collaborator.model_validate(raw)


def save_collaborator(
    collaborator: Collaborator, registry_dir: Path | None = None
) -> None:
    d = registry_dir or _registry_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{collaborator.id}.yml"
    payload = collaborator.model_dump(mode="json")
    path.write_text(
        yaml.dump(payload, allow_unicode=True, sort_keys=False, width=float("inf"))
    )


def list_collaborators(registry_dir: Path | None = None) -> list[Collaborator]:
    d = registry_dir or _registry_dir()
    if not d.exists():
        return []
    result = []
    for yml in sorted(d.glob("*.yml")):
        raw = yaml.safe_load(yml.read_text()) or {}
        result.append(Collaborator.model_validate(raw))
    return result


def delete_collaborator(
    collaborator_id: str,
    registry_dir: Path | None = None,
    active_song_slugs: list[str] | None = None,
) -> None:
    """Remove the collaborator YAML.

    active_song_slugs: if provided and non-empty, raises ValueError (active work orders exist).
    """
    if active_song_slugs:
        raise ValueError(
            f"Cannot delete '{collaborator_id}': active work orders exist for songs: "
            + ", ".join(active_song_slugs)
        )
    d = registry_dir or _registry_dir()
    path = d / f"{collaborator_id}.yml"
    if not path.exists():
        raise FileNotFoundError(f"No collaborator found with id '{collaborator_id}'")
    path.unlink()
