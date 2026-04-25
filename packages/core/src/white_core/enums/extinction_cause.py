from enum import Enum


class ExtinctionCause(str, Enum):
    """Primary extinction drivers"""

    HABITAT_LOSS = "habitat_loss"
    HABITAT_DEGRADATION = "habitat_degradation"
    HABITAT_DESTRUCTION = "habitat_destruction"
    CLIMATE_CHANGE = "climate_change"
    POLLUTION = "pollution"
    OVERHUNTING = "overhunting"
    OVEREXPLOITATION = "overexploitation"
    OVERFISHING = "overfishing"
    INVASIVE_SPECIES = "invasive_species"
    DISEASE = "disease"
    BYCATCH = "bycatch"
    OCEAN_ACIDIFICATION = "ocean_acidification"
    DEFORESTATION = "deforestation"
    POACHING = "poaching"
    PESTICIDES = "pesticides"
    HUMAN_ENCROACHMENT = "human_encroachment"
    RESOURCE_DEPLETION = "resource_depletion"
    COMBINED = "combined_factors"
    UNKNOWN = "unknown"
