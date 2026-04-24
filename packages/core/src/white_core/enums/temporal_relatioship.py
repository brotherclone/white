from enum import Enum


class TemporalRelationship(str, Enum):
    ACROSS = "spans_across"
    BLEED_IN = "bleeds_in"
    BLEED_OUT = "bleeds_out"
    CONTAINED = "contained"
    MATCH = "exact_match"
    UNKNOWN = "unknown"
