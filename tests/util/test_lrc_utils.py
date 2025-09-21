from app.util.lrc_utils import parse_lrc_time

def test_parse_lrc_time_valid():
    assert parse_lrc_time("[00:28.085]") == 28.085
    assert parse_lrc_time("[01:02.123]") == 62.123
    assert parse_lrc_time("[10:00.000]") == 600.0

def test_parse_lrc_time_invalid_format():
    assert parse_lrc_time("00:28.085") is None
    assert parse_lrc_time("[0:28.085]") is None
    assert parse_lrc_time("[00:28]") is None

def test_parse_lrc_time_non_numeric():
    assert parse_lrc_time("[aa:bb.ccc]") is None

def test_parse_lrc_time_empty():
    assert parse_lrc_time("") is None

