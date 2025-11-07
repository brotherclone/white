from enum import Enum


class BookGenre(str, Enum):
    """
    Genre categories for Light Reading collections.
    1. Occult: For books that are about occult and mystical things
    2. SciFi: For books that are about science fiction and fantasy
    3. Sexploitation: Lured, tantalizing material
    4. Cult: Books about cults and fringe groups and their members
    5. Bildungsroman: Books about growing up and the formative years of life
    6. Noir: Post War novels with dark themes and morally ambiguous characters
    7. Psychedelic: Books about mind expansion, altered states, and consciousness
    """

    OCCULT = "occult"
    SCIFI = "scifi"
    SEXPLOITATION = "sexploitation"
    CULT = "cult"
    BILDUNGSROMAN = "bildungsroman"
    NOIR = "noir"
    PSYCHEDELIC = "psychedelic"
