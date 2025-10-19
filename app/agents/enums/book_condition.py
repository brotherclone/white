from enum import Enum


class BookCondition(str, Enum):

    """Physical condition of the tome"""

    PRISTINE = "pristine"
    GOOD = "good"
    WORN = "worn"
    DAMAGED = "damaged"
    FRAGMENTARY = "fragmentary"
    RECONSTRUCTED = "reconstructed from copies"
    BURNED = "partially burned"
