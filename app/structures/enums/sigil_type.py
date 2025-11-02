from enum import Enum

class SigilType(str, Enum):

    """Different approaches to sigil creation"""

    WORD_METHOD = "word_method"
    PICTORIAL = "pictorial"
    MANTRIC = "mantric"
    ALPHABET_OF_DESIRE = "alphabet_of_desire"