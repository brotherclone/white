from __future__ import annotations

from pathlib import Path

import yaml

from white_diary.diary_entry import DiaryEntry


def _diary_dir(production_dir: Path) -> Path:
    return production_dir / "diary"


def _entry_path(entry_id: str, production_dir: Path) -> Path:
    return _diary_dir(production_dir) / f"{entry_id}.yml"


def write_entry(entry: DiaryEntry, production_dir: Path) -> None:
    _diary_dir(production_dir).mkdir(parents=True, exist_ok=True)
    path = _entry_path(entry.id, production_dir)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(
            entry.model_dump(mode="json"), fh, allow_unicode=True, width=float("inf")
        )


def load_entry(entry_id: str, production_dir: Path) -> DiaryEntry:
    path = _entry_path(entry_id, production_dir)
    if not path.exists():
        raise FileNotFoundError(f"Diary entry not found: {entry_id}")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return DiaryEntry.model_validate(data)


def list_entries(production_dir: Path) -> list[DiaryEntry]:
    diary_dir = _diary_dir(production_dir)
    if not diary_dir.exists():
        return []
    entries = []
    for path in diary_dir.glob("*.yml"):
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        entries.append(DiaryEntry.model_validate(data))
    return sorted(entries, key=lambda e: e.created_at)


def delete_entry(entry_id: str, production_dir: Path) -> None:
    path = _entry_path(entry_id, production_dir)
    if not path.exists():
        raise FileNotFoundError(f"Diary entry not found: {entry_id}")
    path.unlink()
