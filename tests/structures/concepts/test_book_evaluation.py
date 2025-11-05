import pytest
from pydantic import ValidationError

from app.structures.concepts.book_evaluation import BookEvaluationDecision


def test_create_valid_model():
    m = BookEvaluationDecision(new_book=True, reaction_book=False, done=False)
    assert isinstance(m, BookEvaluationDecision)
    assert m.new_book is True
    assert m.reaction_book is False
    assert m.done is False
    assert m.model_dump(
        exclude_none=True,
        exclude_unset=True,
    )


def test_missing_fields_raises():
    with pytest.raises(ValidationError):
        BookEvaluationDecision()


def test_wrong_type_raises():
    with pytest.raises(ValidationError):
        BookEvaluationDecision(new_book="yes", reaction_book=False, done=False)


def test_field_descriptions():
    fields = getattr(BookEvaluationDecision, "model_fields", None)
    assert fields is not None
