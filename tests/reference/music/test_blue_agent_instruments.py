from app.reference.music.blue_agent_instruments import (
    BLUE_AGENT_INSTRUMENTS,
    BLUE_AGENT_INSTRUMENTATION_COLOR,
)


def test_instruments_not_empty():
    assert len(BLUE_AGENT_INSTRUMENTS) > 0


def test_instruments_are_strings():
    for instrument in BLUE_AGENT_INSTRUMENTS:
        assert isinstance(instrument, str)
        assert len(instrument) > 0


def test_specific_instruments_present():
    assert "acoustic guitar, Gibson parlor B-25" in BLUE_AGENT_INSTRUMENTS
    assert "mandolin" in BLUE_AGENT_INSTRUMENTS
    assert "banjo" in BLUE_AGENT_INSTRUMENTS


def test_instrumentation_color_not_empty():
    assert len(BLUE_AGENT_INSTRUMENTATION_COLOR) > 0


def test_instrumentation_color_are_strings():
    for color in BLUE_AGENT_INSTRUMENTATION_COLOR:
        assert isinstance(color, str)
        assert len(color) > 0


def test_specific_colors_present():
    assert any("pedal steel" in color for color in BLUE_AGENT_INSTRUMENTATION_COLOR)
    assert any("upright bass" in color for color in BLUE_AGENT_INSTRUMENTATION_COLOR)
