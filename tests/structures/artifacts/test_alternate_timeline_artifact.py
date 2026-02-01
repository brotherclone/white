import pytest
import datetime

from pydantic import ValidationError

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


def create_valid_narrative() -> str:
    """Generate a narrative with 100+ words for testing."""
    return """In this alternate timeline, the decision to board the Greyhound bus to Portland
    instead of returning to New Jersey marked a profound turning point. The city embraced me with
    its perpetual mist and indie spirit, a stark contrast to the suburban landscapes of home.
    Working at Powell's Books became more than just a job; it transformed into a daily immersion
    in literary culture. The smell of aging paper and fresh coffee mingled in the air as I organized
    shelves in the poetry section, discovering voices both familiar and foreign. Rain-slicked streets
    reflected neon signs from cafes where writers gathered late into the night. Weekends meant
    exploring Forest Park's endless trails or catching shows at small venues downtown. The creative
    energy of the Pacific Northwest seeped into my bones, reshaping my understanding of what life
    could be. Each day brought new conversations with customers seeking obscure titles or
    recommendations that led to unexpected connections."""


def create_valid_details() -> list:
    """Generate 5+ specific details for testing."""
    return [
        AlternateLifeDetail(
            category="career",
            detail="Working the poetry section at Powell's",
            sensory_elements=["smell of old books", "rainy Portland streets"],
        ),
        AlternateLifeDetail(
            category="location",
            detail="Renting a studio apartment in the Pearl District",
            sensory_elements=["foggy mornings", "sound of streetcars"],
        ),
        AlternateLifeDetail(
            category="creative",
            detail="Weekly poetry readings at local cafes",
            sensory_elements=["espresso aroma", "intimate venue acoustics"],
        ),
        AlternateLifeDetail(
            category="daily_routine",
            detail="Hiking Forest Park every weekend",
            sensory_elements=["douglas fir scent", "mud on trails"],
        ),
        AlternateLifeDetail(
            category="relationship",
            detail="Forming friendships with local writers and artists",
            sensory_elements=["warm conversations", "collaborative energy"],
        ),
    ]


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
        age_range=(20, 25),
    )

    m = AlternateTimelineArtifact(
        period=period,
        title="Summer in Portland, 1998",
        narrative=create_valid_narrative(),
        divergence_point=divergence,
        key_differences=["Living in Portland", "Working at bookstore"],
        specific_details=create_valid_details(),
        emotional_tone=QuantumTapeEmotionalTone.NOSTALGIC,
        mood_description="Wistful and creative",
        preceding_events=["Graduated college"],
        following_events=["Returned to NJ"],
    )

    assert isinstance(m, AlternateTimelineArtifact)
    assert m.title == "Summer in Portland, 1998"
    assert m.emotional_tone == QuantumTapeEmotionalTone.NOSTALGIC
    assert len(m.key_differences) == 2
    assert len(m.specific_details) == 5
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
        age_range=(20, 25),
    )

    m = AlternateTimelineArtifact(
        period=period,
        title="Test",
        narrative=create_valid_narrative(),
        divergence_point=divergence,
        key_differences=["test"],
        specific_details=create_valid_details(),
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
        age_range=(20, 25),
    )

    with pytest.raises(ValidationError):
        AlternateTimelineArtifact(
            period=period,
            title="Test",
            narrative=create_valid_narrative(),
            divergence_point=divergence,
            key_differences=[],
            specific_details=create_valid_details(),
            emotional_tone=QuantumTapeEmotionalTone.MELANCHOLY,
            mood_description="test",
            preceding_events=[],
            following_events=[],
            plausibility_score=1.5,  # Invalid: > 1.0
        )


def test_field_descriptions():
    fields = getattr(AlternateTimelineArtifact, "model_fields", None)
    assert fields is not None


def test_for_prompt_event_bullets_align():
    divergence = DivergencePoint(
        when="After graduating college in 1997",
        what_changed="Took the Greyhound to Portland instead of returning to NJ",
        why_plausible="Had been offered a job at Powell's Books",
    )

    period = BiographicalPeriod(
        start_date=datetime.date(1997, 6, 1),
        end_date=datetime.date(1998, 8, 31),
        description="Working at Powell's Books in Portland",
        age_range=(20, 25),
    )

    m = AlternateTimelineArtifact(
        period=period,
        title="Alignment Test",
        narrative=create_valid_narrative(),
        divergence_point=divergence,
        key_differences=["Living in Portland", "Working at bookstore"],
        specific_details=create_valid_details(),
        emotional_tone=QuantumTapeEmotionalTone.NOSTALGIC,
        mood_description="Wistful",
        preceding_events=["Graduated college", "Moved out of dorm"],
        following_events=["Returned to NJ"],
    )

    output = m.for_prompt()
    lines = output.splitlines()

    event_lines = [ln for ln in lines if "⍿" in ln]
    assert len(event_lines) == 3  # two preceding + one following

    # All event lines should have the same leading-space count (aligned)
    leading_space_counts = [len(ln) - len(ln.lstrip(" ")) for ln in event_lines]
    assert len(set(leading_space_counts)) == 1

    # Ensure specific text from inputs appears with the event bullet
    assert any("⍿ Graduated college" in ln for ln in event_lines)
    assert any("⍿ Returned to NJ" in ln for ln in event_lines)


def test_for_prompt_renders_alternate_life_detail_detail_field():
    divergence = DivergencePoint(when="test", what_changed="test", why_plausible="test")
    period = BiographicalPeriod(
        start_date=datetime.date(1997, 6, 1),
        end_date=datetime.date(1998, 8, 31),
        description="test",
        age_range=(20, 25),
    )

    detail_obj = AlternateLifeDetail(
        category="creative",
        detail="Unique detail text for rendering",
        sensory_elements=["scent"],
    )

    m = AlternateTimelineArtifact(
        period=period,
        title="Detail Render Test",
        narrative=create_valid_narrative(),
        divergence_point=divergence,
        key_differences=["KD"],
        specific_details=[detail_obj, detail_obj, detail_obj, detail_obj, detail_obj],
        emotional_tone=QuantumTapeEmotionalTone.MELANCHOLY,
        mood_description="test",
        preceding_events=[],
        following_events=[],
    )

    output = m.for_prompt()
    assert "Unique detail text for rendering" in output
    detail_line = next(
        (ln for ln in output.splitlines() if "Unique detail text for rendering" in ln),
        None,
    )
    assert detail_line is not None
    assert detail_line.lstrip().startswith("✧")


def test_format_items_empty_returns_empty_string():
    divergence = DivergencePoint(when="test", what_changed="test", why_plausible="test")
    period = BiographicalPeriod(
        start_date=datetime.date(1997, 6, 1),
        end_date=datetime.date(1998, 8, 31),
        description="test",
        age_range=(20, 25),
    )

    m = AlternateTimelineArtifact(
        period=period,
        title="Empty Format Test",
        narrative=create_valid_narrative(),
        divergence_point=divergence,
        key_differences=["KD"],
        specific_details=create_valid_details(),
        emotional_tone=QuantumTapeEmotionalTone.MELANCHOLY,
        mood_description="test",
        preceding_events=[],
        following_events=[],
    )

    assert m._format_items([], "⍿") == ""
