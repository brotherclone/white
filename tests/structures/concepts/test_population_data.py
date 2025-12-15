from app.structures.concepts.population_data import PopulationData


def test_population_data():
    tp = PopulationData(year=3333, population=0)
    assert tp.population == 0
    assert tp.year == 3333
    assert tp.source == "estimated"
    assert tp.confidence == "medium"
    assert tp.model_dump(exclude_none=True, exclude_unset=True)
