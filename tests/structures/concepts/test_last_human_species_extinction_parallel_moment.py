from pydantic import BaseModel

from app.structures.concepts.last_human_species_extinction_parallel_moment import (
    LastHumanSpeciesExtinctionParallelMoment,
)


def test_inheritance():
    assert issubclass(LastHumanSpeciesExtinctionParallelMoment, BaseModel)


# ToDo: Add additional tests for LastHumanSpeciesExtinctionParallelMoment fields and validation logic
