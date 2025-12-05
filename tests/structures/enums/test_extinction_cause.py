import enum
import pytest

from app.structures.enums.extinction_cause import ExtinctionCause

EXPECTED = {
    "HABITAT_LOSS": "habitat_loss",
    "CLIMATE_CHANGE": "climate_change",
    "POLLUTION": "pollution",
    "OVERHUNTING": "overhunting",
    "INVASIVE_SPECIES": "invasive_species",
    "DISEASE": "disease",
    "BYCATCH": "bycatch",
    "OCEAN_ACIDIFICATION": "ocean_acidification",
    "COMBINED": "combined_factors",
}


def test_members_and_values():
    assert set(ExtinctionCause.__members__.keys()) == set(EXPECTED.keys())
    for name, expected_value in EXPECTED.items():
        member = getattr(ExtinctionCause, name)
        assert member.value == expected_value
        assert isinstance(member.value, str)


def test_members_are_str_and_enum_and_compare_to_value():
    for member in ExtinctionCause:
        assert isinstance(member, ExtinctionCause)
        assert isinstance(member, str)
        assert isinstance(member, enum.Enum)
        assert member == member.value


@pytest.mark.parametrize(
    "value,member",
    [
        ("habitat_loss", ExtinctionCause.HABITAT_LOSS),
        ("climate_change", ExtinctionCause.CLIMATE_CHANGE),
        ("pollution", ExtinctionCause.POLLUTION),
        ("overhunting", ExtinctionCause.OVERHUNTING),
        ("invasive_species", ExtinctionCause.INVASIVE_SPECIES),
        ("disease", ExtinctionCause.DISEASE),
        ("bycatch", ExtinctionCause.BYCATCH),
        ("ocean_acidification", ExtinctionCause.OCEAN_ACIDIFICATION),
        ("combined_factors", ExtinctionCause.COMBINED),
    ],
)
def test_lookup_by_value(value, member):
    assert ExtinctionCause(value) is member


def test_lookup_by_name():
    assert ExtinctionCause["OCEAN_ACIDIFICATION"] is ExtinctionCause.OCEAN_ACIDIFICATION
    assert ExtinctionCause["CLIMATE_CHANGE"] is ExtinctionCause.CLIMATE_CHANGE


def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        ExtinctionCause("dancing")


def test_values_are_unique():
    values = [m.value for m in ExtinctionCause]
    assert len(values) == len(set(values))
