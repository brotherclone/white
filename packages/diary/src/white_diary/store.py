from __future__ import annotations

from pathlib import Path

import yaml

from white_diary.diary_entry import DiaryEntry


def _entry_path(entry_id: str, diary_dir: Path) -> Path:
    return diary_dir / f"{entry_id}.yml"


def write_entry(entry: DiaryEntry, diary_dir: Path) -> None:
    diary_dir.mkdir(parents=True, exist_ok=True)
    path = _entry_path(entry.id, diary_dir)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(
            entry.model_dump(mode="json"), fh, allow_unicode=True, width=float("inf")
        )


def load_entry(entry_id: str, diary_dir: Path) -> DiaryEntry:
    path = _entry_path(entry_id, diary_dir)
    if not path.exists():
        raise FileNotFoundError(f"Diary entry not found: {entry_id}")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return DiaryEntry.model_validate(data)


def list_entries(diary_dir: Path) -> list[DiaryEntry]:
    if not diary_dir.exists():
        return []
    entries = []
    for path in diary_dir.glob("*.yml"):
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        entries.append(DiaryEntry.model_validate(data))
    return sorted(entries, key=lambda e: e.created_at)


def delete_entry(entry_id: str, diary_dir: Path) -> None:
    path = _entry_path(entry_id, diary_dir)
    if not path.exists():
        raise FileNotFoundError(f"Diary entry not found: {entry_id}")
    path.unlink()
