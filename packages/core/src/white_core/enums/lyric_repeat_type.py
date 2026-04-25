from enum import Enum


class LyricRepeatType(str, Enum):
    """How a repeated melody loop's lyrics relate to the first occurrence.

    EXACT       — lyrics are written once and copied verbatim to every
                  repetition (chorus, refrain, hook).
    EXACT_REPEAT — internal marker for subsequent occurrences of an EXACT
                  section; the prompt skips generation and reuses the first.
    VARIATION   — each instance gets its own lines but shares rhyme scheme,
                  meter, and core imagery (verse 2 vs. verse 1).
    FRESH       — each instance is fully independent (bridge, outro, climax).
    """

    EXACT = "exact"
    EXACT_REPEAT = "exact_repeat"
    VARIATION = "variation"
    FRESH = "fresh"
