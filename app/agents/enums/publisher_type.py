from enum import Enum


class PublisherType(str, Enum):

    """Types of publishers for forbidden knowledge"""

    UNIVERSITY = "University imprint"
    OCCULT = "Occult cottage industry publisher"
    SAMIZDAT = "Samizdat dead drop"
    VANITY = "Vanity press directly from author"
    LOST = "Previously deemed lost"
    GOVERNMENT = "Declassified document from FOIA request"