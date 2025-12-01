import pytest
from pydantic import ValidationError

from app.structures.concepts.game_evaluation import GameEvaluationDecision


def test_create_valid_model():
    m = GameEvaluationDecision(should_add_to_story=False)
    assert isinstance(m, GameEvaluationDecision)
    assert m.should_add_to_story is False
    assert m.model_dump(exclude_none=True, exclude_unset=True)


def test_missing_fields_raises():
    with pytest.raises(ValidationError):
        GameEvaluationDecision()


def test_wrong_type_raises():
    with pytest.raises(ValidationError):
        GameEvaluationDecision(should_add_to_story="yes")


def test_field_descriptions():
    fields = getattr(GameEvaluationDecision, "model_fields", None)
    assert fields is not None
