from enum import Enum


class ArchiveStatus(str, Enum):
    ARCHIVED = "archived"
    RESTORED = "restored"
