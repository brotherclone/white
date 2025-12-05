from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.artifacts.last_human_species_extinction_narative_artifact import (
    LastHumanSpeciesExtinctionNarrativeArtifact,
)


def test_inheritance():
    assert issubclass(LastHumanSpeciesExtinctionNarrativeArtifact, ChainArtifact)


# ToDo: Add additional tests for LastHumanSpeciesExtinctionNarrativeArtifact fields and validation logic
