from enum import Enum


class PublisherType(str, Enum):
    """Types of publishers for forbidden knowledge"""

    UNIVERSITY = "university"
    OCCULT = "occult"
    SAMIZDAT = "samizdat"
    VANITY = "vanity"
    LOST = "lost"
    GOVERNMENT = "government"
