REALITY_CORRECTIONS = [
    "actually",
    "really",
    "truly",
    "literally",
    "genuinely",
    "in fact",
    "in reality",
    "in truth",
]

# Substitution and replacement
# "Instead of X" = what-could-have-been, alternative realities
SUBSTITUTION_MARKERS = ["instead", "rather", "rather than", "not", "but"]

# Category revision signals
# "Different from X" = boundary redrawing, category violation
CATEGORY_REVISION = [
    "different",
    "unlike",
    "contrary to",
    "opposite",
    "contrast",
    "versus",
    "compared to",
]

# Ontological uncertainty markers
# "Seemed like X" = IMAGINED bleeding into REAL
ONTOLOGICAL_UNCERTAINTY = [
    "seemed",
    "appeared",
    "felt like",
    "looked like",
    "sounded like",
    "as if",
    "as though",
    "like",
    "supposedly",
    "apparently",
    "presumably",
    "allegedly",
]

# Transformation and state change
# "Turned out to be X" = ontological revelation
TRANSFORMATION_MARKERS = [
    "turned out",
    "became",
    "transformed",
    "shifted",
    "changed into",
    "evolved into",
    "converted",
    "morphed",
    "transmuted",
]

# Discovery and realization
# "Realized it was X" = correcting internal model
DISCOVERY_MARKERS = [
    "realized",
    "discovered",
    "found out",
    "learned",
    "understood",
    "recognized",
    "saw that",
    "knew that",
    "thought",
    "believed",
]

# Negation patterns (challenging assumed categories)
# "Wasn't what I thought" = category violation discovery
NEGATION_PATTERNS = [
    "wasn't",
    "weren't",
    "isn't",
    "aren't",
    "didn't",
    "don't",
    "not what",
    "not really",
    "not exactly",
    "not quite",
]

# Liminal/boundary category markers
# "Neither X nor Y", "both X and Y" = refusing binary categorization
LIMINAL_MARKERS = [
    "neither",
    "nor",
    "both",
    "and",
    "not quite",
    "sort of",
    "kind of",
    "more like",
    "less like",
    "between",
    "somewhere between",
    "partly",
    "partially",
    "somewhat",
    "half",
]

# All rebracketing words combined
REBRACKETING_WORDS = (
    REALITY_CORRECTIONS
    + SUBSTITUTION_MARKERS
    + CATEGORY_REVISION
    + ONTOLOGICAL_UNCERTAINTY
    + TRANSFORMATION_MARKERS
    + DISCOVERY_MARKERS
    + NEGATION_PATTERNS
    + LIMINAL_MARKERS
)
