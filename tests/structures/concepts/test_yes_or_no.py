import pytest
from pydantic import ValidationError

from app.structures.concepts.yes_or_no import YesOrNo


def test_yes_or_no():
    """Test YesOrNo with True value"""
    yes = YesOrNo(answer=True)
    assert yes.answer is True


def test_yes_or_no_false():
    """Test YesOrNo with False value"""
    no = YesOrNo(answer=False)
    assert no.answer is False


def test_yes_or_no_wrong_type():
    """Test that the answer must be boolean"""
    with pytest.raises(ValidationError):
        YesOrNo(answer={"invalid": "dict"})
