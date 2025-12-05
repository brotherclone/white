from pydantic import BaseModel

from app.structures.concepts.extinction_musical_parameters import (
    ExtinctionMusicalParameters,
)


def test_inheritance():
    assert issubclass(ExtinctionMusicalParameters, BaseModel)


# ToDo: Add additional tests for ExtinctionMusicalParameters fields and validation logic
# ToDo: Test for alignment to SongProposal
