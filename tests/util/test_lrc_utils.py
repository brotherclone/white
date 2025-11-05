from app.util import lrc_utils as lu


def test_parse_lrc_time_valid():
    assert abs(lu.parse_lrc_time("[00:28.085]") - 28.085) < 1e-6
    assert abs(lu.parse_lrc_time("[01:02.123]") - 62.123) < 1e-6
    assert abs(lu.parse_lrc_time("[10:00.000]") - 600.0) < 1e-6


def test_parse_lrc_time_invalid():
    assert lu.parse_lrc_time("[bad]") is None


def test_smpte_to_lrc_timestamp_valid():
    # hh:mm:ss:ff.sub -> convert where ff are frames at default fps=30
    out = lu.smpte_to_lrc_timestamp("00:01:02:15.0", fps=30)
    assert out == "[01:02.500]"


def test_convert_file_smpte_to_lrc_lines():
    lines = ["00:00:01:00.0 some text", "No time here", "12:00:00:10.5"]
    out = lu.convert_file_smpte_to_lrc_lines(lines, fps=30)
    # first line should be converted to an LRC timestamp starting with '['
    assert out[0].startswith("[")
    assert out[1] == "No time here"
    # third line not matching pattern remains unchanged


def test_load_lrc_parsing_basic(tmp_path):
    content = """
[ti:Song Title]
[00:00.000]
Intro
[00:05.000] First line
[00:10.000] Second line

[00:15.000] Third line
""".strip()
    p = tmp_path / "test.lrc"
    p.write_text(content, encoding="utf-8")

    lyrics = lu.load_lrc(str(p))
    # Expect 4 lyrical entries: Intro, First line, Second line, Third line
    assert len(lyrics) == 4
    assert lyrics[0]["text"] == "Intro"
    assert abs(lyrics[0]["start_time"] - 0.0) < 1e-6
    assert lyrics[1]["text"] == "First line"
    assert abs(lyrics[1]["start_time"] - 5.0) < 1e-6
    # end_time of first equals start_time of second
    assert abs(lyrics[0]["end_time"] - lyrics[1]["start_time"]) < 1e-6
    # last end_time is start_time + 3.0
    assert abs(lyrics[-1]["end_time"] - (lyrics[-1]["start_time"] + 3.0)) < 1e-6


def test_load_lrc_file_not_found():
    assert lu.load_lrc("/path/does/not/exist.lrc") == []
