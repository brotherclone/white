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
    UNKNOWN = "unknown"
