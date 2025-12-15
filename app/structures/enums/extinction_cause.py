from enum import Enum


class ExtinctionCause(str, Enum):
    """Primary extinction drivers"""

    HABITAT_LOSS = "habitat_loss"
    CLIMATE_CHANGE = "climate_change"
    POLLUTION = "pollution"
    OVERHUNTING = "overhunting"
    INVASIVE_SPECIES = "invasive_species"
    DISEASE = "disease"
    BYCATCH = "bycatch"
    OCEAN_ACIDIFICATION = "ocean_acidification"
    COMBINED = "combined_factors"
