from pathlib import Path

from white_diary.diary_entry import DiaryEntry
from white_diary.store import delete_entry, list_entries, load_entry, write_entry

ENTRIES_DIR: Path = Path(__file__).parent.parent / "entries"

__all__ = [
    "DiaryEntry",
    "ENTRIES_DIR",
    "write_entry",
    "load_entry",
    "list_entries",
    "delete_entry",
]
