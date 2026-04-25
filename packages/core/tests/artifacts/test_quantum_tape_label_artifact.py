from pathlib import Path

import pytest
from pydantic import ValidationError

from white_core.artifacts.quantum_tape_label_artifact import (
    QuantumTapeLabelArtifact,
)
from white_core.enums.quantum_tape_recording_quality import (
    QuantumTapeRecordingQuality,
)


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


# --- New tests added below ---


def test_for_prompt_original_label_visibility():
    # When original_label_visible is True and original_label_text present
    m1 = QuantumTapeLabelArtifact(
        title="T1",
        date_range="D1",
        recording_quality=QuantumTapeRecordingQuality.SP,
        counter_start=0,
        original_label_visible=True,
        original_label_text="Left on label",
    )
    p1 = m1.for_prompt()
    assert "Title: T1" in p1
    assert "Date Range: D1" in p1
    assert "Original Label Text: Left on label" in p1

    # When original_label_visible is True but original_label_text is None -> line still present but empty
    m2 = QuantumTapeLabelArtifact(
        title="T2",
        date_range="D2",
        recording_quality=QuantumTapeRecordingQuality.SP,
        counter_start=0,
        original_label_visible=True,
        original_label_text=None,
    )
    p2 = m2.for_prompt()
    assert "Title: T2" in p2
    assert "Date Range: D2" in p2
    # The prompt will include the field with a None, rendered as 'None' or empty string depending on model_dump substitution; accept either
    assert "Original Label Text" in p2

    # When original_label_visible is False the original label line should not be present
    m3 = QuantumTapeLabelArtifact(
        title="T3",
        date_range="D3",
        recording_quality=QuantumTapeRecordingQuality.SP,
        counter_start=0,
        original_label_visible=False,
        original_label_text="ShouldNotAppear",
    )
    p3 = m3.for_prompt()
    assert "Title: T3" in p3
    assert "Date Range: D3" in p3
    assert "Original Label Text" not in p3


def _make_minimal(title="Test Title", **kwargs):
    return QuantumTapeLabelArtifact(
        title=title,
        date_range="1998-06 to 1999-11",
        recording_quality=QuantumTapeRecordingQuality.SP,
        counter_start=0,
        **kwargs,
    )


def test_artifact_name_derived_from_title():
    m = _make_minimal(title="Summer in Portland 1998")
    assert "UNKNOWN" not in (m.file_name or "")
    assert "UNKNOWN" not in m.artifact_name


def test_artifact_name_explicit_override():
    m = _make_minimal(title="Some Title", artifact_name="my_override")
    assert m.artifact_name == "my_override"


def test_save_file_writes_yml(tmp_path):
    base = tmp_path / "out"
    base.mkdir()

    m = _make_minimal(
        base_path=str(base),
        year_documented="1998",
        original_date="1998",
        original_title="Gabe Walsh — 1998",
        tapeover_date="Jun 1998 – Nov 1999",
        tapeover_title="The Lost Portland Chapter",
        subject_name="Gabe Walsh",
        age_during="22–23",
        location="Portland, OR",
        catalog_number="QT-B-1998-ABCDEF",
    )
    m.save_file()

    content = Path(m.get_artifact_path()).read_text(encoding="utf-8")
    for value in [
        "Gabe Walsh",
        "Portland, OR",
        "QT-B-1998-ABCDEF",
    ]:
        assert value in content, f"Expected '{value}' in saved YAML"


def test_flatten_includes_template_fields():
    m = _make_minimal(
        year_documented="2001",
        original_date="2001",
        original_title="Gabe Walsh — 2001",
        tapeover_date="Jan 2001 – Dec 2002",
        tapeover_title="Alternate 2001",
        subject_name="Gabe Walsh",
        age_during="25–27",
        location="Brooklyn, NY",
        catalog_number="QT-B-2001-ZZZTOP",
    )
    flat = m.flatten()
    for key in [
        "year_documented",
        "original_date",
        "original_title",
        "tapeover_date",
        "tapeover_title",
        "subject_name",
        "age_during",
        "location",
        "catalog_number",
    ]:
        assert key in flat, f"Expected '{key}' in flatten() output"
    assert flat["subject_name"] == "Gabe Walsh"
    assert flat["catalog_number"] == "QT-B-2001-ZZZTOP"


def test_save_file_writes_valid_yaml(tmp_path):
    base = tmp_path / "out"
    base.mkdir()

    m = QuantumTapeLabelArtifact(
        title="SaveTest",
        date_range="2020",
        recording_quality=QuantumTapeRecordingQuality.SP,
        counter_start=0,
        base_path=str(base),
        original_label_visible=True,
        original_label_text=None,
    )
    m.save_file()

    import yaml

    out_file = Path(m.get_artifact_path(with_file_name=True))
    assert out_file.exists()
    data = yaml.safe_load(out_file.read_text(encoding="utf-8"))
    assert data["title"] == "SaveTest"
    assert data["date_range"] == "2020"
    assert data["original_label_text"] is None
