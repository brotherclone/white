from enum import Enum


class ChainArtifactType(str, Enum):

    EVP_ARTIFACT = "evp_artifact"
    INSTRUCTIONS_TO_HUMAN = "instructions_to_human"
    SIGIL = "sigil_description"
    BOOK = "book"
    NEWSPAPER_ARTICLE = "newspaper_article"
    SYMBOLIC_OBJECT = "symbolic_object"
    PROPOSAL = "proposal"
    GAME_RUN = "game_run"
    CHARACTER_SHEET = "character_sheet"
    CHARACTER_PORTRAIT = "character_portrait"
    ARBITRARYS_SURVEY = "arbitrary_survey"
    LAST_HUMAN = "last_human"
    LAST_HUMAN_SPECIES_EXTINCTION_NARRATIVE = "last_human_species_extinction_narrative"
    SPECIES_EXTINCTION = "species_extinction"
    RESCUE_DECISION = "rescue_decision"
    QUANTUM_TAPE_LABEL = "quantum_tape_label"
    ALTERNATE_TIMELINE = "alternate_timeline"
    UNKNOWN = "unknown"
