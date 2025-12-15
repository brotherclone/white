# Basic temporal prepositions and conjunctions
TEMPORAL_PREPOSITIONS = [
    "when",
    "before",
    "after",
    "during",
    "while",
    "then",
    "now",
    "past",
    "since",
    "until",
    "till",
]

# Time units and measures
TEMPORAL_UNITS = [
    "year",
    "month",
    "week",
    "day",
    "hour",
    "minute",
    "second",
    "moment",
    "instant",
    "season",
    "decade",
    "century",
    "era",
    "age",
    "time",
    "morning",
    "afternoon",
    "evening",
    "night",
    "noon",
    "midnight",
    "dawn",
    "dusk",
    "twilight",
]

# Relative temporal markers
TEMPORAL_RELATIVE = [
    "yesterday",
    "today",
    "tomorrow",
    "tonight",
    "ago",
    "later",
    "earlier",
    "soon",
    "recently",
    "formerly",
    "previously",
    "currently",
    "presently",
    "eventually",
    "finally",
]

# Frequency and duration
TEMPORAL_FREQUENCY = [
    "always",
    "never",
    "sometimes",
    "often",
    "rarely",
    "seldom",
    "occasionally",
    "frequently",
    "constantly",
    "continually",
    "once",
    "twice",
    "thrice",
    "again",
    "anymore",
]

# Temporal sequence
TEMPORAL_SEQUENCE = [
    "first",
    "second",
    "third",
    "last",
    "next",
    "previous",
    "prior",
    "following",
    "preceding",
    "initial",
    "final",
    "meanwhile",
    "simultaneously",
    "concurrent",
    "sequential",
    "successive",
]

# Temporal states (ongoing vs completed)
TEMPORAL_STATES = [
    "already",
    "yet",
    "still",
    "just",
    "begun",
    "ended",
    "finished",
    "started",
    "completed",
    "ongoing",
    "continuing",
    "lasting",
]

# Temporal modifiers
TEMPORAL_MODIFIERS = [
    "brief",
    "long",
    "short",
    "quick",
    "slow",
    "sudden",
    "gradual",
    "immediate",
    "delayed",
    "prompt",
    "swift",
    "prolonged",
    "temporary",
    "permanent",
    "eternal",
    "forever",
    "fleeting",
]

# All temporal words combined
TEMPORAL_WORDS = (
    TEMPORAL_PREPOSITIONS
    + TEMPORAL_UNITS
    + TEMPORAL_RELATIVE
    + TEMPORAL_FREQUENCY
    + TEMPORAL_SEQUENCE
    + TEMPORAL_STATES
    + TEMPORAL_MODIFIERS
)

# Regex patterns for temporal deixis (memory anchors)
# These catch phrases like "the year was '93", "back in 2010", "my first summer"
TEMPORAL_DEIXIS_PATTERNS = [
    r"\bthe year was\b",
    r"\bback in\b",
    r"\bin \d{4}\b",  # "in 1993", "in '93"
    r"\bin \'\d{2}\b",  # "in '93"
    r"\bthat \w+ in\b",  # "that summer in", "that day in"
    r"\bused to\b",
    r"\bwould always\b",
    r"\bnever forgot\b",
    r"\bstill recall\b",
    r"\bcan still\b",
    r"\bmy \w+ year\b",  # "my first year", "my senior year"
    r"\bour first\b",
    r"\bthat summer\b",
    r"\bthat winter\b",
    r"\bthat spring\b",
    r"\bthat fall\b",
    r"\bthose days when\b",
    r"\b(before|after|since|until) (the|that|we|I)\b",
]
