from app.reference.rebracketing_words.wish_fulfillment_words import (
    WISH_FULFILLMENT_WORDS,
)


def test_wish_fulfillment_words_not_empty():
    assert len(WISH_FULFILLMENT_WORDS) > 0


def test_wish_fulfillment_words_are_strings():
    for word in WISH_FULFILLMENT_WORDS:
        assert isinstance(word, str)
        assert len(word) > 0


def test_specific_words_present():
    assert "famous" in WISH_FULFILLMENT_WORDS
    assert "wealthy" in WISH_FULFILLMENT_WORDS
    assert "rockstar" in WISH_FULFILLMENT_WORDS
