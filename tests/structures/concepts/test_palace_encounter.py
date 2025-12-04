import pytest
from pydantic import ValidationError

from app.structures.concepts.pulsar_palace_encounter import PulsarPalaceEncounter


def test_create_valid_model():
    m = PulsarPalaceEncounter(encounter_id="test-id")
    assert isinstance(m, PulsarPalaceEncounter)
    assert m.encounter_id == "test-id"
    assert m.model_dump(exclude_none=True) == {"encounter_id": "test-id"}


def test_missing_fields_raises():
    with pytest.raises(ValidationError):
        PulsarPalaceEncounter()


def test_wrong_type_raises():
    with pytest.raises(ValidationError):
        PulsarPalaceEncounter(narrative=40.1)


def test_field_descriptions():
    fields = getattr(PulsarPalaceEncounter, "model_fields", None)
    assert fields is not None
