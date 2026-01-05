from app.reference.rebracketing_words.fantasy_genre_words import FANTASY_GENRE_WORDS


def test_fantasy_genre_words_not_empty():
    assert len(FANTASY_GENRE_WORDS) > 0


def test_fantasy_genre_words_are_strings():
    for word in FANTASY_GENRE_WORDS:
        assert isinstance(word, str)
        assert len(word) > 0


def test_specific_words_present():
    assert "magic" in FANTASY_GENRE_WORDS
    assert "alien" in FANTASY_GENRE_WORDS
    assert "time travel" in FANTASY_GENRE_WORDS
