from enum import Enum

class BookGenre(str, Enum):

    """Genre categories for Light Reading collection"""

    OCCULT = "occult"
    SCIFI = "scifi"
    SEXPLOITATION = "sexploitation"
    CULT = "cult"
    BILDUNGSROMAN = "bildungsroman"
    NOIR = "noir"
    PSYCHEDELIC = "psychedelic"