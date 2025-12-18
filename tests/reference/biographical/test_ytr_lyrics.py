from app.reference.biographical.ytr_lyrics import YOUR_TEAM_RING_TAPE_LYRIC_FRAGMENTS


def test_lyric_fragments_not_empty():
    assert len(YOUR_TEAM_RING_TAPE_LYRIC_FRAGMENTS) > 0


def test_lyric_fragments_are_strings():
    for fragment in YOUR_TEAM_RING_TAPE_LYRIC_FRAGMENTS:
        assert isinstance(fragment, str)
        assert len(fragment) > 0


def test_specific_fragments_present():
    assert "We have nine ways" in YOUR_TEAM_RING_TAPE_LYRIC_FRAGMENTS
    assert "salute the waves" in YOUR_TEAM_RING_TAPE_LYRIC_FRAGMENTS
