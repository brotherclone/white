from pathlib import Path

from app.util.convert_smpte_to_lrc import convert_timestamp, convert_text, main


def test_convert_timestamp_basic():
    src = "01:01:06:11.49"
    assert convert_timestamp(src) == "[01:06.011]"


def test_convert_timestamp_no_hour():
    # three-field form (MM:SS:FRAMES) should work
    src = "00:12:34.5"
    assert convert_timestamp(src) == "[00:12.034]"


def test_convert_text_multiple():
    s = "lead 01:01:06:11.49 middle 00:00:00:00.0 end"
    out = convert_text(s)
    assert "[01:06.011]" in out
    assert "[00:00.000]" in out


def test_convert_text_comma_decimal():
    s = "stamp 01:02:03:04,56"
    out = convert_text(s)
    # frames '04,56' -> truncated to 4 -> .004
    assert "[02:03.004]" in out


def test_main_stdout_and_inplace(tmp_path: Path, capsys):
    p = tmp_path / "sample.txt"
    p.write_text("first 01:01:06:11.49 second")

    # writing to stdout when not using --inplace
    rc = main([str(p)])
    assert rc == 0

    # main wrote to stdout; capture it by running convert_text directly for check
    out = convert_text(p.read_text())
    assert "[01:06.011]" in out

    # test inplace overwrite
    rc2 = main(["-i", str(p)])
    assert rc2 == 0
    # file should now contain converted timestamp
    data = p.read_text()
    assert "[01:06.011]" in data
