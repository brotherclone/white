import enum

import pytest

from app.structures.enums.player import RainbowPlayer

EXPECTED = {
    "GABE": "Gabriel Walsh",
    "JOSH": "Josh Plotner",
    "REMEZ": "Remez",
    "MARVIN": "Marvin Muoneké",
    "GRAHAM": "Graham Hopkins",
    "MARIA": "Maria Grigoryeva",
    "LYUDMILA": "Lyudmila Kadyrbaeva",
    "ZOLTAN": "Zoltan Renaldi",
}


def test_members_and_values():
    assert set(RainbowPlayer.__members__.keys()) == set(EXPECTED.keys())
    for name, expected_value in EXPECTED.items():
        member = getattr(RainbowPlayer, name)
        assert member.value == expected_value
        assert isinstance(member.value, str)


def test_members_are_str_and_enum_and_compare_to_value():
    for member in RainbowPlayer:
        assert isinstance(member, RainbowPlayer)
        assert isinstance(member, str)
        assert isinstance(member, enum.Enum)
        assert member == member.value


@pytest.mark.parametrize(
    "value,member",
    [
        ("Gabriel Walsh", RainbowPlayer.GABE),
        ("Josh Plotner", RainbowPlayer.JOSH),
        ("Remez", RainbowPlayer.REMEZ),
    ],
)
def test_lookup_by_value(value, member):
    assert RainbowPlayer(value) is member


def test_lookup_by_name():
    assert RainbowPlayer["GABE"] is RainbowPlayer.GABE
    assert RainbowPlayer["JOSH"] is RainbowPlayer.JOSH


def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        RainbowPlayer("Unknown")


def test_values_are_unique():
    values = [m.value for m in RainbowPlayer]
    assert len(values) == len(set(values))


def test_enum_members_are_enum_instances():
    assert isinstance(RainbowPlayer.GABE, enum.Enum)
    assert isinstance(RainbowPlayer.GRAHAM, enum.Enum)
