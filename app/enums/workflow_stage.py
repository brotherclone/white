from enum import Enum


class WorkflowStage(Enum):
    PLANNING = "planning"
    CHORD_GENERATION = "chord_generation"
    MELODY_GENERATION = "melody_generation"
    LYRICS_GENERATION = "lyrics_generation"
    ARRANGEMENT = "arrangement"
    REFINEMENT = "refinement"
    VALIDATION = "validation"
