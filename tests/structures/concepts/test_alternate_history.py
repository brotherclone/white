import pytest
from pydantic import ValidationError
import datetime

from app.structures.artifacts.alternate_timeline_artifact import (
    AlternateTimelineArtifact,
)
from app.structures.concepts.alternate_life_detail import AlternateLifeDetail
from app.structures.concepts.biographical_period import BiographicalPeriod
from app.structures.concepts.divergence_point import DivergencePoint
from app.structures.enums.quantum_tape_emotional_tone import QuantumTapeEmotionalTone
from app.structures.enums.biographical_timeline_detail_level import (
    BiographicalTimelineDetailLevel,
)


def test_create_valid_model():
    divergence = DivergencePoint(
        when="After graduating college in 1997",
        what_changed="Took the Greyhound to Portland instead of returning to NJ",
        why_plausible="Had been offered a job at Powell's Books",
    )

    period = BiographicalPeriod(
        start_date=datetime.date(1997, 6, 1),
        end_date=datetime.date(1998, 8, 31),
        description="Working at Powell's Books in Portland",
        detail_level=BiographicalTimelineDetailLevel.HIGH,
    )

    detail = AlternateLifeDetail(
        category="career",
        detail="Working the poetry section at Powell's",
        sensory_elements=["smell of old books", "rainy Portland streets"],
    )

    m = AlternateTimelineArtifact(
        period=period,
        title="Summer in Portland, 1998",
        narrative="A full prose description of alternate life",
        divergence_point=divergence,
        key_differences=["Living in Portland", "Working at bookstore"],
        specific_details=[detail],
        emotional_tone=QuantumTapeEmotionalTone.NOSTALGIC,
        mood_description="Wistful and creative",
        preceding_events=["Graduated college"],
        following_events=["Returned to NJ"],
    )

    assert isinstance(m, AlternateTimelineArtifact)
    assert m.title == "Summer in Portland, 1998"
    assert m.emotional_tone == QuantumTapeEmotionalTone.NOSTALGIC
    assert len(m.key_differences) == 2
    assert len(m.specific_details) == 1
    assert m.model_dump(exclude_none=True, exclude_unset=True)


def test_missing_required_fields_raises():
    with pytest.raises(ValidationError):
        AlternateTimelineArtifact()


def test_wrong_type_raises():
    with pytest.raises(ValidationError):
        AlternateTimelineArtifact(
            period="not a period", title="Summer in Portland, 1998"
        )


def test_optional_scores():
    divergence = DivergencePoint(
        when="After graduating college in 1997",
        what_changed="Took the Greyhound to Portland",
        why_plausible="Had been offered a job",
    )

    period = BiographicalPeriod(
        start_date=datetime.date(1997, 6, 1),
        end_date=datetime.date(1998, 8, 31),
        description="Working at Powell's Books",
    )

    m = AlternateTimelineArtifact(
        period=period,
        title="Test",
        narrative="Test narrative",
        divergence_point=divergence,
        key_differences=["test"],
        specific_details=[],
        emotional_tone=QuantumTapeEmotionalTone.MELANCHOLY,
        mood_description="test",
        preceding_events=[],
        following_events=[],
        plausibility_score=0.85,
        specificity_score=0.75,
        divergence_magnitude=0.6,
    )

    assert m.plausibility_score == 0.85
    assert m.specificity_score == 0.75
    assert m.divergence_magnitude == 0.6


def test_score_validation():
    divergence = DivergencePoint(when="test", what_changed="test", why_plausible="test")
    period = BiographicalPeriod(
        start_date=datetime.date(1997, 6, 1),
        end_date=datetime.date(1998, 8, 31),
        description="test",
    )

    with pytest.raises(ValidationError):
        AlternateTimelineArtifact(
            period=period,
            title="Test",
            narrative="Test",
            divergence_point=divergence,
            key_differences=[],
            specific_details=[],
            emotional_tone=QuantumTapeEmotionalTone.MELANCHOLY,
            mood_description="test",
            preceding_events=[],
            following_events=[],
            plausibility_score=1.5,  # Invalid: > 1.0
        )


def test_field_descriptions():
    fields = getattr(AlternateTimelineArtifact, "model_fields", None)
    assert fields is not None
