import pytest
from pydantic import BaseModel, ValidationError

from app.structures.concepts.Interview_item import InterviewItem


def test_init():
    item = InterviewItem(question="What is your favorite color?", answer="Blue")
    assert item.question == "What is your favorite color?"
    assert item.answer == "Blue"
    assert isinstance(item, InterviewItem)
    assert item.model_dump(exclude_none=True, exclude_unset=True) == dict(
        question="What is your favorite color?", answer="Blue"
    )
    assert issubclass(InterviewItem, BaseModel)


def test_init_without_answer():
    """Test initialization with only question (answer is optional)"""
    item = InterviewItem(question="What is your name?")
    assert item.question == "What is your name?"
    assert item.answer is None


def test_init_with_none_answer():
    """Test explicit None answer"""
    item = InterviewItem(question="Test?", answer=None)
    assert item.question == "Test?"
    assert item.answer is None


def test_required_question():
    """Test that question is required"""
    with pytest.raises(ValidationError):
        InterviewItem(answer="An answer without a question")

    with pytest.raises(ValidationError):
        InterviewItem()


def test_empty_answer():
    """Test with empty string answer"""
    item = InterviewItem(question="Empty?", answer="")
    assert item.question == "Empty?"
    assert item.answer == ""


def test_long_question_and_answer():
    """Test with lengthy question and answer"""
    long_question = (
        "What are your thoughts on the philosophical implications of consciousness in artificial intelligence?"
        * 10
    )
    long_answer = "This is a complex topic..." * 50

    item = InterviewItem(question=long_question, answer=long_answer)
    assert item.question == long_question
    assert item.answer == long_answer


def test_model_dump():
    """Test model_dump with and without None values"""
    item = InterviewItem(question="Q?")
    dump_with_none = item.model_dump()
    dump_exclude_none = item.model_dump(exclude_none=True)

    assert dump_with_none == {"question": "Q?", "answer": None}
    assert dump_exclude_none == {"question": "Q?"}


def test_model_dump_json():
    """Test JSON serialization"""
    item = InterviewItem(question="Test question", answer="Test answer")
    json_str = item.model_dump_json()
    assert "Test question" in json_str
    assert "Test answer" in json_str


def test_equality():
    """Test equality comparison"""
    item1 = InterviewItem(question="Q1", answer="A1")
    item2 = InterviewItem(question="Q1", answer="A1")
    item3 = InterviewItem(question="Q1", answer="A2")
    item4 = InterviewItem(question="Q2", answer="A1")

    assert item1 == item2
    assert item1 != item3
    assert item1 != item4


def test_special_characters_in_question_and_answer():
    """Test with special characters and formatting"""
    item = InterviewItem(
        question='What\'s the "meaning" of life?',
        answer="It's 42! (According to Douglas Adams)",
    )
    assert item.question == 'What\'s the "meaning" of life?'
    assert item.answer == "It's 42! (According to Douglas Adams)"


def test_multiline_answer():
    """Test with multiline answer"""
    item = InterviewItem(question="Describe yourself", answer="Line 1\nLine 2\nLine 3")
    assert "\n" in item.answer
    assert item.answer.count("\n") == 2


def test_update_answer():
    """Test updating answer after creation"""
    item = InterviewItem(question="Test?")
    assert item.answer is None

    # Pydantic models are immutable by default, but we can create a copy
    updated_item = item.model_copy(update={"answer": "New answer"})
    assert updated_item.answer == "New answer"
    assert item.answer is None  # Original should be unchanged
