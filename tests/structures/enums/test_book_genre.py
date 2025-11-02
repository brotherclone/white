import enum
import pytest
from app.structures.enums.book_genre import BookGenre

EXPECTED = {
    "OCCULT": "occult",
    "SCIFI": "scifi",
    "SEXPLOITATION": "sexploitation",
    "CULT": "cult",
    "BILDUNGSROMAN": "bildungsroman",
    "NOIR": "noir",
    "PSYCHEDELIC": "psychedelic",
}


def test_members_and_values():
    assert set(BookGenre.__members__.keys()) == set(EXPECTED.keys())
    for name, expected_value in EXPECTED.items():
        member = getattr(BookGenre, name)
        assert member.value == expected_value
        assert isinstance(member.value, str)


@pytest.mark.parametrize("value,member", [
    ("occult", BookGenre.OCCULT),
    ("scifi", BookGenre.SCIFI),
    ("sexploitation", BookGenre.SEXPLOITATION),
    ("cult", BookGenre.CULT),
    ("bildungsroman", BookGenre.BILDUNGSROMAN),
    ("noir", BookGenre.NOIR),
    ("psychedelic", BookGenre.PSYCHEDELIC),
])
def test_lookup_by_value(value, member):
    assert BookGenre(value) is member


def test_lookup_by_name():
    assert BookGenre["OCCULT"] is BookGenre.OCCULT
    assert BookGenre["SCIFI"] is BookGenre.SCIFI


def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        BookGenre("romcom")


def test_values_are_unique():
    values = [m.value for m in BookGenre]
    assert len(values) == len(set(values))


def test_enum_members_are_enum_instances():
    assert isinstance(BookGenre.OCCULT, enum.Enum)
    assert isinstance(BookGenre.NOIR, enum.Enum)
