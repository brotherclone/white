import pytest
from pydantic import ValidationError

from app.structures.concepts.pulsar_palace_character import PulsarPalaceCharacter


def test_create_valid_model():
    m = PulsarPalaceCharacter(encounter_id="test-id", thread_id="test-thread")
    assert isinstance(m, PulsarPalaceCharacter)
    assert m.encounter_id == "test-id"
    assert m.thread_id == "test-thread"
    assert m.model_dump(exclude_none=True) == {
        "encounter_id": "test-id",
        "thread_id": "test-thread",
    }


def test_missing_fields_raises():
    with pytest.raises(ValidationError):
        PulsarPalaceCharacter()


def test_wrong_type_raises():
    with pytest.raises(ValidationError):
        PulsarPalaceCharacter(narrative=40.1)


def test_field_descriptions():
    fields = getattr(PulsarPalaceCharacter, "model_fields", None)
    assert fields is not None
