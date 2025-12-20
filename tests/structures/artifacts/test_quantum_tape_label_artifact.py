import pytest
from pydantic import ValidationError

from app.structures.artifacts.quantum_tape_label_artifact import (
    QuantumTapeLabelArtifact,
)
from app.structures.enums.quantum_tape_recording_quality import (
    QuantumTapeRecordingQuality,
)

# ToDo: Add for_prompt() tests


def test_create_valid_model():
    m = QuantumTapeLabelArtifact(
        title="Summer in Portland - 1998",
        date_range="1998-06 to 1999-11",
        recording_quality=QuantumTapeRecordingQuality.SP,
        counter_start=0,
    )
    assert isinstance(m, QuantumTapeLabelArtifact)
    assert m.title == "Summer in Portland - 1998"
    assert m.date_range == "1998-06 to 1999-11"
    assert m.recording_quality == QuantumTapeRecordingQuality.SP
    assert m.counter_start == 0
    assert m.counter_end is None
    assert m.notes is None
    assert m.original_label_visible is True
    assert m.original_label_text is None
    assert m.tape_degradation is None
    assert m.tape_brand == "TASCAM 424-S"
    assert m.handwriting_style is None
    assert m.model_dump(exclude_none=True, exclude_unset=True)


def test_with_all_optional_fields():
    m = QuantumTapeLabelArtifact(
        title="Test Tape",
        date_range="1997-01 to 1997-12",
        recording_quality=QuantumTapeRecordingQuality.EP,
        counter_start=1234,
        counter_end=5678,
        notes="Wrote novel. Didn't show anyone.",
        original_label_visible=False,
        original_label_text="Gabe Walsh",
        tape_degradation=0.25,
        tape_brand="Sony T-120",
        handwriting_style="cursive",
    )
    assert m.counter_end == 5678
    assert m.notes == "Wrote novel. Didn't show anyone."
    assert m.original_label_visible is False
    assert m.original_label_text == "Gabe Walsh"
    assert m.tape_degradation == 0.25
    assert m.tape_brand == "Sony T-120"
    assert m.handwriting_style == "cursive"


def test_missing_required_fields_raises():
    with pytest.raises(ValidationError):
        QuantumTapeLabelArtifact()

    with pytest.raises(ValidationError):
        QuantumTapeLabelArtifact(title="test")

    with pytest.raises(ValidationError):
        QuantumTapeLabelArtifact(title="test", date_range="1997-01 to 1997-12")


def test_wrong_type_raises():
    with pytest.raises(ValidationError):
        QuantumTapeLabelArtifact(
            title=123,
            date_range="1997-01 to 1997-12",
            recording_quality=QuantumTapeRecordingQuality.SP,
            counter_start=0,
        )


def test_counter_start_validation():
    # Valid boundaries
    m1 = QuantumTapeLabelArtifact(
        title="test",
        date_range="test",
        recording_quality=QuantumTapeRecordingQuality.SP,
        counter_start=0,
    )
    assert m1.counter_start == 0

    m2 = QuantumTapeLabelArtifact(
        title="test",
        date_range="test",
        recording_quality=QuantumTapeRecordingQuality.SP,
        counter_start=9999,
    )
    assert m2.counter_start == 9999

    # Invalid: too low
    with pytest.raises(ValidationError):
        QuantumTapeLabelArtifact(
            title="test",
            date_range="test",
            recording_quality=QuantumTapeRecordingQuality.SP,
            counter_start=-1,
        )

    # Invalid: too high
    with pytest.raises(ValidationError):
        QuantumTapeLabelArtifact(
            title="test",
            date_range="test",
            recording_quality=QuantumTapeRecordingQuality.SP,
            counter_start=10000,
        )


def test_counter_end_validation():
    # Valid boundaries
    m1 = QuantumTapeLabelArtifact(
        title="test",
        date_range="test",
        recording_quality=QuantumTapeRecordingQuality.SP,
        counter_start=0,
        counter_end=0,
    )
    assert m1.counter_end == 0

    m2 = QuantumTapeLabelArtifact(
        title="test",
        date_range="test",
        recording_quality=QuantumTapeRecordingQuality.SP,
        counter_start=0,
        counter_end=9999,
    )
    assert m2.counter_end == 9999

    # Invalid: too low
    with pytest.raises(ValidationError):
        QuantumTapeLabelArtifact(
            title="test",
            date_range="test",
            recording_quality=QuantumTapeRecordingQuality.SP,
            counter_start=0,
            counter_end=-1,
        )

    # Invalid: too high
    with pytest.raises(ValidationError):
        QuantumTapeLabelArtifact(
            title="test",
            date_range="test",
            recording_quality=QuantumTapeRecordingQuality.SP,
            counter_start=0,
            counter_end=10000,
        )


def test_tape_degradation_validation():
    # Valid boundaries
    m1 = QuantumTapeLabelArtifact(
        title="test",
        date_range="test",
        recording_quality=QuantumTapeRecordingQuality.SP,
        counter_start=0,
        tape_degradation=0.0,
    )
    assert m1.tape_degradation == 0.0

    m2 = QuantumTapeLabelArtifact(
        title="test",
        date_range="test",
        recording_quality=QuantumTapeRecordingQuality.SP,
        counter_start=0,
        tape_degradation=1.0,
    )
    assert m2.tape_degradation == 1.0
    with pytest.raises(ValidationError):
        QuantumTapeLabelArtifact(
            title="test",
            date_range="test",
            recording_quality=QuantumTapeRecordingQuality.SP,
            counter_start=0,
            tape_degradation=-0.1,
        )
    with pytest.raises(ValidationError):
        QuantumTapeLabelArtifact(
            title="test",
            date_range="test",
            recording_quality=QuantumTapeRecordingQuality.SP,
            counter_start=0,
            tape_degradation=1.5,
        )


def test_default_tape_brand():
    m = QuantumTapeLabelArtifact(
        title="test",
        date_range="test",
        recording_quality=QuantumTapeRecordingQuality.SP,
        counter_start=0,
    )
    assert m.tape_brand == "TASCAM 424-S"


def test_default_original_label_visible():
    m = QuantumTapeLabelArtifact(
        title="test",
        date_range="test",
        recording_quality=QuantumTapeRecordingQuality.SP,
        counter_start=0,
    )
    assert m.original_label_visible is True


def test_field_descriptions():
    fields = getattr(QuantumTapeLabelArtifact, "model_fields", None)
    assert fields is not None
