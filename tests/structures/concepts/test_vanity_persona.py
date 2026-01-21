import pytest
from pydantic import BaseModel

from app.structures.concepts.vanity_persona import VanityPersona
from app.structures.enums.vanity_interviewer_type import VanityInterviewerType


@pytest.fixture
def make_persona():
    """Factory fixture to create VanityPersona instances with sensible defaults.

    Usage:
        p = make_persona()  # default Alice Smith, hostile_skeptical
        p = make_persona(first_name="Jane", last_name="Doe")
    """

    def _make(**overrides):
        defaults = dict(
            first_name="Alice",
            last_name="Smith",
            publication="MocknRoll",
            interviewer_type=VanityInterviewerType.HOSTILE_SKEPTICAL,
        )
        defaults.update(overrides)
        return VanityPersona(**defaults)

    return _make


def test_init(make_persona):
    p = make_persona()
    assert p.first_name == "Alice"
    assert p.last_name == "Smith"
    assert p.interviewer_type == VanityInterviewerType.HOSTILE_SKEPTICAL
    dumped = p.model_dump(exclude_none=True)
    assert dumped["first_name"] == "Alice"
    assert dumped["last_name"] == "Smith"
    assert dumped["interviewer_type"] == "hostile_skeptical"
    assert dumped["stance"] == "This is pretentious nonsense"
    assert dumped["approach"] == "Aggressive, reductive, gotcha-focused"
    assert dumped["goal"] == "Make artist look stupid or contradictory"
    assert dumped["publication"] == "MocknRoll"
    assert "tactics" in dumped and isinstance(dumped["tactics"], list)
    assert "example_questions" in dumped and isinstance(
        dumped["example_questions"], list
    )

    assert isinstance(p, VanityPersona)
    assert isinstance(p, BaseModel)


def test_init_with_different_names(make_persona):
    """Test initialization with various name combinations"""
    p = make_persona(first_name="John", last_name="Doe")
    assert p.first_name == "John"
    assert p.last_name == "Doe"


def test_required_fields():
    """Test that model provides defaults for missing names"""
    p = VanityPersona()
    assert isinstance(p.first_name, str)
    assert isinstance(p.last_name, str)


def test_model_validation(make_persona):
    """Test that model validates field types correctly"""
    p = make_persona(first_name="Test", last_name="User")
    assert p.first_name == "Test"
    p2 = make_persona(first_name="", last_name="")
    assert p2.first_name == ""
    assert p2.last_name == ""


def test_model_dump(make_persona):
    """Test model_dump output"""
    p = make_persona(first_name="Jane", last_name="Doe", publication="MocknRoll")
    dump = p.model_dump()
    assert dump == {
        "first_name": "Jane",
        "last_name": "Doe",
        "interviewer_type": "hostile_skeptical",
        "publication": "MocknRoll",
        "stance": "This is pretentious nonsense",
        "approach": "Aggressive, reductive, gotcha-focused",
        "tactics": [
            "False dichotomies",
            "Reductive readings that strip nuance",
            "Gotcha questions about contradictions",
            "Accusations of pretension",
            "Demand for immediate accessibility",
        ],
        "goal": "Make artist look stupid or contradictory",
        "example_questions": [
            "So you're saying {{concept}} - isn't that just {{reductive_version}}?",
            "This whole {{methodology}} thing - don't you think that's a bit pretentious?",
            "If {{aspect_a}} then how can you also claim {{aspect_b}}? Sounds contradictory.",
            "Can you explain in plain English what this is actually about? Because it sounds like gibberish.",
            "Isn't this just {{established_artist}} but worse?",
        ],
    }


def test_model_dump_json(make_persona):
    """Test JSON serialization"""
    p = make_persona(first_name="Bob", last_name="Builder")
    json_str = p.model_dump_json()
    assert "Bob" in json_str
    assert "Builder" in json_str


def test_equality(make_persona):
    """Test equality comparison"""
    p1 = make_persona(first_name="Alice", last_name="Smith")
    p2 = make_persona(first_name="Alice", last_name="Smith")
    p3 = make_persona(first_name="Bob", last_name="Smith")

    assert p1 == p2
    assert p1 != p3


def test_with_special_characters(make_persona):
    """Test names with special characters"""
    p = make_persona(first_name="José", last_name="O'Brien")
    assert p.first_name == "José"
    assert p.last_name == "O'Brien"
