import pytest

from app.structures.concepts.rainbow_table_color import (
    RainbowColorModes, RainbowColorOntologicalMode,
    RainbowTableTransmigrationalMode, get_rainbow_table_color)


def test_enum_str_and_repr():
    assert str(RainbowColorModes.TIME) == "Temporal"
    assert "RainbowColorModes.TIME" in repr(RainbowColorModes.TIME)


def test_transmigrational_mode_str_and_repr():
    m = RainbowTableTransmigrationalMode(
        current_mode=RainbowColorModes.SPACE,
        transitory_mode=RainbowColorModes.TIME,
        transcendental_mode=RainbowColorModes.INFORMATION,
    )
    s = str(m)
    assert "->" in s
    r = repr(m)
    assert "RainbowTableTransmigrationalMode" in r


def test_rainbow_table_color_to_dict_and_get_rgba():
    # pick a known color from the table
    color = get_rainbow_table_color("R")
    d = color.to_dict()
    assert d["color_name"] == "Red"
    assert isinstance(d["hex_value"], int) or isinstance(d["hex_value"], int)
    rgba = color.get_rgba()
    assert isinstance(rgba, tuple) and len(rgba) == 4
    r, g, b, a = rgba
    # basic range checks
    assert 0 <= r <= 255
    assert 0 <= g <= 255
    assert 0 <= b <= 255
    assert a == 255


def test_get_rainbow_table_color_invalid():
    with pytest.raises(ValueError):
        get_rainbow_table_color("Z_NOT_PRESENT")


def test_ontological_mode_str_and_repr():
    mode = RainbowColorOntologicalMode.KNOWN
    assert str(mode) == "Known"
    assert "RainbowColorOntologicalMode.KNOWN" in repr(mode)
