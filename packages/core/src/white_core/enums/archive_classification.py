from enum import Enum


class ArchiveClassification(str, Enum):
    PLACED = "placed"
    KEPT = "kept"
    CANDIDATE = "candidate"
    UNMATCHED = "unmatched"
