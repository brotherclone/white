from enum import Enum


class RebracketingAnalysisType(str, Enum):
    CAUSAL = "causal"
    SPATIAL = "spatial"
    PERCEPTUAL = "perceptual"
    EXPERIENTIAL = "experiential"
    TEMPORAL = "temporal"
    BOUNDARY = "boundary"
    NONE = "none"
