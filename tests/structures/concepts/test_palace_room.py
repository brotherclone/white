import pytest
from pydantic import ValidationError

from app.structures.concepts.pulsar_palace_room import PulsarPalaceRoom


def test_create_valid_model():
    m = PulsarPalaceRoom(room_id="test-id")
    assert isinstance(m, PulsarPalaceRoom)
    assert m.room_id == "test-id"
    assert m.model_dump(exclude_none=True) == {"room_id": "test-id"}


def test_missing_fields_raises():
    with pytest.raises(ValidationError):
        PulsarPalaceRoom()


def test_wrong_type_raises():
    with pytest.raises(ValidationError):
        PulsarPalaceRoom(description=40.1)


def test_field_descriptions():
    fields = getattr(PulsarPalaceRoom, "model_fields", None)
    assert fields is not None
