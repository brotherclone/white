import pytest
from pydantic import ValidationError

from app.structures.concepts.multimodal_segment import MultimodalSegment


def test_create_valid_model_minimal():
    m = MultimodalSegment(
        manifest_id="test-manifest-001",
        segment_type="section",
        segment_id="seg-001",
        canonical_start=0.0,
        canonical_end=30.5,
        duration=30.5,
        section_name="Intro",
    )
    assert isinstance(m, MultimodalSegment)
    assert m.manifest_id == "test-manifest-001"
    assert m.segment_type == "section"
    assert m.segment_id == "seg-001"
    assert m.canonical_start == 0.0
    assert m.canonical_end == 30.5
    assert m.duration == 30.5
    assert m.section_name == "Intro"
    assert m.lyrical_content == []
    assert m.mood_tags == []
    assert m.boundary_crossing_indicators == []
    assert m.model_dump(exclude_none=True, exclude_unset=True)


def test_create_with_musical_context():
    m = MultimodalSegment(
        manifest_id="test-001",
        segment_type="section",
        segment_id="seg-001",
        canonical_start=0.0,
        canonical_end=30.0,
        duration=30.0,
        section_name="Verse",
        bpm=120,
        time_signature="4/4",
        key="C Major",
        rainbow_color="R",
    )
    assert m.bpm == 120
    assert m.time_signature == "4/4"
    assert m.key == "C Major"
    assert m.rainbow_color == "R"


def test_create_with_lyrical_content():
    lyrics = [
        {"text": "First line", "timestamp": 0.5},
        {"text": "Second line", "timestamp": 2.0},
    ]

    m = MultimodalSegment(
        manifest_id="test-001",
        segment_type="section",
        segment_id="seg-001",
        canonical_start=0.0,
        canonical_end=30.0,
        duration=30.0,
        section_name="Verse",
        lyrical_content=lyrics,
    )
    assert len(m.lyrical_content) == 2
    assert m.lyrical_content[0]["text"] == "First line"


def test_create_with_rebracketing_scores():
    m = MultimodalSegment(
        manifest_id="test-001",
        segment_type="section",
        segment_id="seg-001",
        canonical_start=0.0,
        canonical_end=30.0,
        duration=30.0,
        section_name="Verse",
        rebracketing_score=0.75,
        memory_discrepancy_severity=0.6,
        temporal_complexity=0.8,
        section_rebracketing_score=0.7,
        comprehensive_rebracketing_score=0.72,
    )
    assert m.rebracketing_score == 0.75
    assert m.memory_discrepancy_severity == 0.6
    assert m.temporal_complexity == 0.8
    assert m.section_rebracketing_score == 0.7
    assert m.comprehensive_rebracketing_score == 0.72


def test_missing_required_fields_raises():
    with pytest.raises(ValidationError):
        MultimodalSegment()

    with pytest.raises(ValidationError):
        MultimodalSegment(manifest_id="test")


def test_wrong_type_raises():
    with pytest.raises(ValidationError):
        MultimodalSegment(
            manifest_id=123,
            segment_type="section",
            segment_id="seg-001",
            canonical_start=0.0,
            canonical_end=30.0,
            duration=30.0,
            section_name="Verse",
        )


def test_negative_time_values_raise():
    with pytest.raises(ValidationError):
        MultimodalSegment(
            manifest_id="test",
            segment_type="section",
            segment_id="seg-001",
            canonical_start=-1.0,
            canonical_end=30.0,
            duration=30.0,
            section_name="Verse",
        )


def test_score_validation_boundaries():
    # Valid boundaries
    m = MultimodalSegment(
        manifest_id="test",
        segment_type="section",
        segment_id="seg-001",
        canonical_start=0.0,
        canonical_end=30.0,
        duration=30.0,
        section_name="Verse",
        rebracketing_score=0.0,
    )
    assert m.rebracketing_score == 0.0

    m2 = MultimodalSegment(
        manifest_id="test",
        segment_type="section",
        segment_id="seg-001",
        canonical_start=0.0,
        canonical_end=30.0,
        duration=30.0,
        section_name="Verse",
        rebracketing_score=1.0,
    )
    assert m2.rebracketing_score == 1.0

    # Invalid: > 1.0
    with pytest.raises(ValidationError):
        MultimodalSegment(
            manifest_id="test",
            segment_type="section",
            segment_id="seg-001",
            canonical_start=0.0,
            canonical_end=30.0,
            duration=30.0,
            section_name="Verse",
            rebracketing_score=1.5,
        )


def test_to_dict_method():
    m = MultimodalSegment(
        manifest_id="test-001",
        segment_type="section",
        segment_id="seg-001",
        canonical_start=0.0,
        canonical_end=30.0,
        duration=30.0,
        section_name="Verse",
    )

    d = m.to_dict()
    assert isinstance(d, dict)
    assert d["manifest_id"] == "test-001"
    assert d["segment_type"] == "section"


def test_from_dict_classmethod():
    data = {
        "manifest_id": "test-001",
        "segment_type": "section",
        "segment_id": "seg-001",
        "canonical_start": 0.0,
        "canonical_end": 30.0,
        "duration": 30.0,
        "section_name": "Verse",
    }

    m = MultimodalSegment.from_dict(data)
    assert isinstance(m, MultimodalSegment)
    assert m.manifest_id == "test-001"
    assert m.section_name == "Verse"


def test_with_audio_features():
    m = MultimodalSegment(
        manifest_id="test",
        segment_type="section",
        segment_id="seg-001",
        canonical_start=0.0,
        canonical_end=30.0,
        duration=30.0,
        section_name="Verse",
        audio_features={"tempo": 120, "energy": 0.8},
        audio_tracks_features=[
            {"player": "guitar", "energy": 0.7},
            {"player": "drums", "energy": 0.9},
        ],
    )
    assert m.audio_features["tempo"] == 120
    assert len(m.audio_tracks_features) == 2


def test_with_players_info():
    m = MultimodalSegment(
        manifest_id="test",
        segment_type="section",
        segment_id="seg-001",
        canonical_start=0.0,
        canonical_end=30.0,
        duration=30.0,
        section_name="Verse",
        players=["guitar", "bass", "drums"],
        player_count=3,
    )
    assert len(m.players) == 3
    assert "guitar" in m.players
    assert m.player_count == 3


def test_with_metadata():
    m = MultimodalSegment(
        manifest_id="test",
        segment_type="section",
        segment_id="seg-001",
        canonical_start=0.0,
        canonical_end=30.0,
        duration=30.0,
        section_name="Verse",
        title="My Song",
        mood_tags=["melancholic", "introspective"],
        concept="Memory and loss",
        section_description="Opening verse about childhood",
    )
    assert m.title == "My Song"
    assert len(m.mood_tags) == 2
    assert "melancholic" in m.mood_tags
    assert m.concept == "Memory and loss"
    assert m.section_description == "Opening verse about childhood"


def test_boundary_crossing_indicators():
    m = MultimodalSegment(
        manifest_id="test",
        segment_type="section",
        segment_id="seg-001",
        canonical_start=0.0,
        canonical_end=30.0,
        duration=30.0,
        section_name="Verse",
        boundary_crossing_indicators=["temporal_bleed", "modal_shift"],
    )
    assert len(m.boundary_crossing_indicators) == 2
    assert "temporal_bleed" in m.boundary_crossing_indicators


def test_field_descriptions():
    fields = getattr(MultimodalSegment, "model_fields", None)
    assert fields is not None
