from enum import Enum


class BookCondition(str, Enum):

    """
    Physical condition of the tome
    1. Pristine: As new, flawless condition
    2. Good: As good as new, minor wear
    3. Worn: As worn out, but fully readable
    4. Damaged: As damaged, sections destroyed by physical damage
    5. Fragmentary: As fragmentary, only parts remain
    6. Reconstructed: As reconstructed, but not complete
    7. Burned: As burned, sections destroyed by fire
    """

    PRISTINE = "pristine"
    GOOD = "good"
    WORN = "worn"
    DAMAGED = "damaged"
    FRAGMENTARY = "fragmentary"
    RECONSTRUCTED = "reconstructed from copies"
    BURNED = "partially burned"
