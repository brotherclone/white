import pytest

from app.structures.music.core.lyric import LyricFile, LyricPair


def test_lyric_pair_valid():
    lp = LyricPair(lyric="Hello", timestamp="[00:01.000]")
    assert lp.lyric == "Hello"
    assert lp.timestamp == "[00:01.000]"


def test_lyric_pair_missing_fields():
    with pytest.raises(ValueError):
        LyricPair(lyric="", timestamp="[00:01.000]")
    with pytest.raises(ValueError):
        LyricPair(lyric="Hello", timestamp="")


def test_lyric_file_valid():
    lp = LyricPair(lyric="Hello", timestamp="[00:01.000]")
    lf = LyricFile(title="Song", artist="Artist", album="Album", lyrics=[lp])
    assert lf.title == "Song"
    assert lf.lyrics[0].lyric == "Hello"
