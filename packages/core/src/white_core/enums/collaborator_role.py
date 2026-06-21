from enum import Enum


class CollaboratorRole(str, Enum):
    VOCALIST = "vocalist"
    DRUMMER = "drummer"
    GUITARIST = "guitarist"
    BASSIST = "bassist"
    KEYS = "keys"
    STRINGS = "strings"
    BRASS = "brass"
    MIXING = "mixing"
    MASTERING = "mastering"
    OTHER = "other"
