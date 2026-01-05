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


def test_extract_lyrics_from_lrc_basic(tmp_path):
    """Test extracting lyrics from LRC file."""
    content = """[ti:Test Song]
[ar:Test Artist]
[00:00.000] First line
[00:05.000] Second line
[00:10.000] Third line
"""
    p = tmp_path / "lyrics.lrc"
    p.write_text(content, encoding="utf-8")

    lyrics = lu.extract_lyrics_from_lrc(str(p))
    assert lyrics is not None
    assert "First line" in lyrics
    assert "Second line" in lyrics
    assert "Third line" in lyrics
    # Timestamps should be removed
    assert "[00:00.000]" not in lyrics
    assert "[ti:Test Song]" not in lyrics
    assert "[ar:Test Artist]" not in lyrics


def test_extract_lyrics_from_lrc_file_not_found():
    """Test extract_lyrics_from_lrc with non-existent file."""
    result = lu.extract_lyrics_from_lrc("/path/does/not/exist.lrc")
    assert result is None


def test_extract_lyrics_from_lrc_empty_file(tmp_path):
    """Test extract_lyrics_from_lrc with empty file."""
    p = tmp_path / "empty.lrc"
    p.write_text("", encoding="utf-8")

    result = lu.extract_lyrics_from_lrc(str(p))
    assert result is None


def test_extract_lyrics_from_lrc_only_metadata(tmp_path):
    """Test extract_lyrics_from_lrc with only metadata tags."""
    content = """[ti:Test Song]
[ar:Test Artist]
[al:Test Album]
[by:Test Creator]
"""
    p = tmp_path / "metadata_only.lrc"
    p.write_text(content, encoding="utf-8")

    result = lu.extract_lyrics_from_lrc(str(p))
    # Should return None since there's no actual lyrical content
    assert result is None


def test_parse_lrc_time_edge_cases():
    """Test parse_lrc_time with various formats."""
    # Valid cases
    assert abs(lu.parse_lrc_time("[00:00.000]") - 0.0) < 1e-6
    assert abs(lu.parse_lrc_time("[99:59.999]") - 5999.999) < 1e-6

    # Invalid cases
    assert lu.parse_lrc_time("00:00.000") is None  # Missing brackets
    assert lu.parse_lrc_time("[0:0.0]") is None  # Wrong format
    assert lu.parse_lrc_time("") is None  # Empty string
    assert lu.parse_lrc_time("invalid") is None  # Invalid format


def test_smpte_to_lrc_timestamp_edge_cases():
    """Test smpte_to_lrc_timestamp with various inputs."""
    # Basic conversion
    assert lu.smpte_to_lrc_timestamp("00:00:00:00.0", fps=30) == "[00:00.000]"

    # Hour conversion
    assert lu.smpte_to_lrc_timestamp("01:00:00:00.0", fps=30) == "[60:00.000]"

    # Invalid format should return unchanged
    assert lu.smpte_to_lrc_timestamp("invalid") == "invalid"
    assert lu.smpte_to_lrc_timestamp("00:00:00") == "00:00:00"


def test_convert_file_smpte_to_lrc_lines_mixed_content():
    """Test convert_file_smpte_to_lrc_lines with mixed content."""
    lines = [
        "00:00:00:15.0 Line with timecode",
        "Plain text line",
        "01:02:03:00.0 Another timecode line",
        "",
        "More plain text",
    ]
    result = lu.convert_file_smpte_to_lrc_lines(lines, fps=30)

    # First line should be converted to LRC format
    assert result[0].startswith("[")
    # Second line should be unchanged
    assert result[1] == "Plain text line"
    # Third line should be converted
    assert result[2].startswith("[")
    # Empty and plain text should remain
    assert result[3] == ""
    assert result[4] == "More plain text"


def test_load_lrc_with_metadata_tags(tmp_path):
    """Test load_lrc skips metadata tags correctly."""
    content = """[ti:Song Title]
[ar:Artist Name]
[al:Album Name]
[by:Creator]
[offset:+500]
[00:00.000] First lyric
[00:05.000] Second lyric
"""
    p = tmp_path / "with_metadata.lrc"
    p.write_text(content, encoding="utf-8")

    lyrics = lu.load_lrc(str(p))
    # Should only have 2 lyrical entries, metadata should be skipped
    assert len(lyrics) == 2
    assert lyrics[0]["text"] == "First lyric"
    assert lyrics[1]["text"] == "Second lyric"


def test_load_lrc_inline_text(tmp_path):
    """Test load_lrc with inline text (timestamp and text on same line)."""
    content = """[00:00.000] Inline text one
[00:05.000] Inline text two
"""
    p = tmp_path / "inline.lrc"
    p.write_text(content, encoding="utf-8")

    lyrics = lu.load_lrc(str(p))
    assert len(lyrics) == 2
    assert lyrics[0]["text"] == "Inline text one"
    assert lyrics[1]["text"] == "Inline text two"
    assert abs(lyrics[0]["start_time"] - 0.0) < 1e-6
    assert abs(lyrics[1]["start_time"] - 5.0) < 1e-6
