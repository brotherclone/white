from app.structures.artifacts.species_extinction_artifact import (
    SpeciesExtinctionArtifact,
)
from app.structures.artifacts.base_artifact import ChainArtifact


def test_inheritance():
    assert issubclass(SpeciesExtinctionArtifact, ChainArtifact)


# ToDo: Add additional tests for SpeciesExtinctionArtifact fields and validation logic
