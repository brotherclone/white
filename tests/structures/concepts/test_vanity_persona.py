import pytest
from pydantic import BaseModel, ValidationError

from app.structures.concepts.vanity_persona import VanityPersona


def test_init():
    p = VanityPersona(first_name="Alice", last_name="Smith")
    assert p.first_name == "Alice"
    assert p.last_name == "Smith"
    assert p.model_dump(exclude_none=True) == dict(
        first_name="Alice", last_name="Smith"
    )
    assert isinstance(p, VanityPersona)
    assert isinstance(p, BaseModel)


def test_init_with_different_names():
    """Test initialization with various name combinations"""
    p = VanityPersona(first_name="John", last_name="Doe")
    assert p.first_name == "John"
    assert p.last_name == "Doe"


def test_required_fields():
    """Test that first_name and last_name are required"""
    with pytest.raises(ValidationError):
        VanityPersona(first_name="Alice")

    with pytest.raises(ValidationError):
        VanityPersona(last_name="Smith")

    with pytest.raises(ValidationError):
        VanityPersona()


def test_model_validation():
    """Test that model validates field types correctly"""
    # Should work with valid strings
    p = VanityPersona(first_name="Test", last_name="User")
    assert p.first_name == "Test"

    # Test with empty strings (should be allowed)
    p2 = VanityPersona(first_name="", last_name="")
    assert p2.first_name == ""
    assert p2.last_name == ""


def test_model_dump():
    """Test model_dump output"""
    p = VanityPersona(first_name="Jane", last_name="Doe")
    dump = p.model_dump()
    assert dump == {"first_name": "Jane", "last_name": "Doe"}


def test_model_dump_json():
    """Test JSON serialization"""
    p = VanityPersona(first_name="Bob", last_name="Builder")
    json_str = p.model_dump_json()
    assert "Bob" in json_str
    assert "Builder" in json_str


def test_equality():
    """Test equality comparison"""
    p1 = VanityPersona(first_name="Alice", last_name="Smith")
    p2 = VanityPersona(first_name="Alice", last_name="Smith")
    p3 = VanityPersona(first_name="Bob", last_name="Smith")

    assert p1 == p2
    assert p1 != p3


def test_with_special_characters():
    """Test names with special characters"""
    p = VanityPersona(first_name="José", last_name="O'Brien")
    assert p.first_name == "José"
    assert p.last_name == "O'Brien"
