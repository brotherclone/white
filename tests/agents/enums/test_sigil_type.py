import pytest
from app.agents.enums.sigil_type import SigilType

def test_members_and_values():
    expected = {
        "WORD_METHOD": "word_method",
        "PICTORIAL": "pictorial",
        "MANTRIC": "mantric",
        "ALPHABET_OF_DESIRE": "alphabet_of_desire",
    }
    for name, value in expected.items():
        member = SigilType[name]
        assert member.value == value
        assert isinstance(member.value, str)

@pytest.mark.parametrize("value,member", [
    ("word_method", SigilType.WORD_METHOD),
    ("pictorial", SigilType.PICTORIAL),
    ("mantric", SigilType.MANTRIC),
    ("alphabet_of_desire", SigilType.ALPHABET_OF_DESIRE),
])
def test_lookup_by_value(value, member):
    assert SigilType(value) is member

def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        SigilType("invalid")

def test_values_are_unique():
    values = [m.value for m in SigilType]
    assert len(values) == len(set(values))

