import pytest

from app.structures.concepts.rainbow_table_color import (
    RainbowColorModes, RainbowTableColor, RainbowTableTransmigrationalMode,
    get_rainbow_table_color, the_rainbow_table_colors)


def test_get_rainbow_table_color_returns_known_color():
    color = get_rainbow_table_color("R")
    assert isinstance(color, RainbowTableColor)
    assert color.color_name == "Red"
    assert color.mnemonic_character_value == "R"
    assert color.hex_value == 0xAE1E36


def test_get_rainbow_table_color_invalid_raises():
    with pytest.raises(ValueError):
        get_rainbow_table_color("X")  # not present in the table


def test_enum_str_and_repr():
    assert str(RainbowColorModes.TIME) == "Temporal"
    assert "RainbowColorModes.TIME" in repr(RainbowColorModes.TIME)


def test_transmigrational_mode_str_for_Z():
    # `Z` entry in the table has a transmigrational_mode set
    z = the_rainbow_table_colors["Z"]
    tm = z.transmigrational_mode
    assert isinstance(tm, RainbowTableTransmigrationalMode)
    # values come from the enum's string values
    assert str(tm) == "Objectional -> Temporal -> Ontological"


def test_to_dict_contains_expected_keys_and_values():
    blue = the_rainbow_table_colors["B"]  # Blue has several fields set
    d = blue.to_dict()
    assert d["color_name"] == "Blue"
    assert d["hex_value"] == 0x042A7B
    # to_dict intentionally uses the key `mnemonic_letter_value`
    assert d["mnemonic_letter_value"] == "B"
    assert "transmigrational_mode" in d


def test_get_rgba_computation():
    yellow = the_rainbow_table_colors["Y"]  # 0xFFFF00
    rgba = yellow.get_rgba()
    assert rgba == (255, 255, 0, 255)
