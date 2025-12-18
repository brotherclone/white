import pytest
from pydantic import ValidationError
import datetime

from app.structures.concepts.biographical_period import BiographicalPeriod
from app.structures.concepts.biographical_event import BiographicalEvent
from app.structures.enums.biographical_timeline_detail_level import (
    BiographicalTimelineDetailLevel,
)


def test_create_valid_model():
    m = BiographicalPeriod(
        start_date=datetime.date(1997, 6, 1),
        end_date=datetime.date(1998, 8, 31),
        description="Working at Powell's Books",
        detail_level=BiographicalTimelineDetailLevel.MEDIUM,
        known_events=[],
        key_relationships=[],
        creative_output=[],
        age_range=(20, 25),
    )
    assert isinstance(m, BiographicalPeriod)
    assert m.start_date == datetime.date(1997, 6, 1)
    assert m.end_date == datetime.date(1998, 8, 31)
    assert m.description == "Working at Powell's Books"
    assert m.detail_level == BiographicalTimelineDetailLevel.MEDIUM
    assert m.known_events == []
    assert m.key_relationships == []
    assert m.creative_output == []
    assert m.model_dump(exclude_none=True, exclude_unset=True)


def test_with_events():
    event = BiographicalEvent(date=datetime.date(1997, 7, 1), description="Started job")
    m = BiographicalPeriod(
        start_date=datetime.date(1997, 6, 1),
        end_date=datetime.date(1998, 8, 31),
        description="Working period",
        known_events=[event],
        age_range=(20, 25),
    )
    assert len(m.known_events) == 1
    assert m.known_events[0].description == "Started job"


def test_with_all_optional_fields():
    m = BiographicalPeriod(
        start_date=datetime.date(1997, 6, 1),
        end_date=datetime.date(1998, 8, 31),
        description="College years",
        detail_level=BiographicalTimelineDetailLevel.HIGH,
        location="Portland, OR",
        primary_activity="studying",
        key_relationships=["roommate", "girlfriend"],
        creative_output=["poems", "short stories"],
        emotional_tone="optimistic",
        trauma_level="low",
        age_range=(20, 25),
    )
    assert m.location == "Portland, OR"
    assert m.primary_activity == "studying"
    assert len(m.key_relationships) == 2
    assert len(m.creative_output) == 2
    assert m.emotional_tone == "optimistic"
    assert m.trauma_level == "low"


def test_missing_required_fields_raises():
    with pytest.raises(ValidationError):
        BiographicalPeriod()

    with pytest.raises(ValidationError):
        BiographicalPeriod(start_date=datetime.date(1997, 6, 1))

    with pytest.raises(ValidationError):
        BiographicalPeriod(
            start_date=datetime.date(1997, 6, 1), end_date=datetime.date(1998, 8, 31)
        )


def test_wrong_type_raises():
    with pytest.raises(ValidationError):
        BiographicalPeriod(
            start_date="not a date",
            end_date=datetime.date(1998, 8, 31),
            description="test",
            detail_level=BiographicalTimelineDetailLevel.MEDIUM,
            known_events=[],
            key_relationships=[],
            creative_output=[],
            age_range=range(20, 25),
        )


def test_duration_months_property():
    m = BiographicalPeriod(
        start_date=datetime.date(1997, 6, 1),
        end_date=datetime.date(1998, 8, 31),
        description="test",
        age_range=(20, 25),
        detail_level=BiographicalTimelineDetailLevel.MEDIUM,
        trauma_level="low",
    )
    assert m.duration_months == 14  # June 1997 to August 1998


def test_duration_months_same_year():
    m = BiographicalPeriod(
        start_date=datetime.date(1997, 1, 1),
        end_date=datetime.date(1997, 6, 30),
        description="test",
        age_range=(20, 25),
        detail_level=BiographicalTimelineDetailLevel.MEDIUM,
        trauma_level="low",
    )
    assert m.duration_months == 5


def test_is_forgotten_property_true():
    m = BiographicalPeriod(
        start_date=datetime.date(1997, 1, 1),
        end_date=datetime.date(1997, 7, 31),
        description="test",
        detail_level=BiographicalTimelineDetailLevel.LOW,
        trauma_level="none",
        age_range=(20, 25),
    )
    assert m.is_forgotten is True


def test_is_forgotten_property_minimal_detail():
    m = BiographicalPeriod(
        start_date=datetime.date(1997, 1, 1),
        end_date=datetime.date(1997, 7, 31),
        description="test",
        detail_level=BiographicalTimelineDetailLevel.MINIMAL,
        trauma_level="low",
        age_range=(20, 25),
    )
    assert m.is_forgotten is True


def test_is_forgotten_property_false_high_detail():
    m = BiographicalPeriod(
        start_date=datetime.date(1997, 1, 1),
        end_date=datetime.date(1997, 7, 31),
        description="test",
        detail_level=BiographicalTimelineDetailLevel.HIGH,
        trauma_level="none",
        age_range=(20, 25),
    )
    assert m.is_forgotten is False


def test_is_forgotten_property_false_high_trauma():
    m = BiographicalPeriod(
        start_date=datetime.date(1997, 1, 1),
        end_date=datetime.date(1997, 7, 31),
        description="test",
        detail_level=BiographicalTimelineDetailLevel.LOW,
        trauma_level="high",
        age_range=(20, 25),
    )
    assert m.is_forgotten is False


def test_is_forgotten_property_false_short_duration():
    m = BiographicalPeriod(
        start_date=datetime.date(1997, 1, 1),
        end_date=datetime.date(1997, 5, 31),
        description="test",
        detail_level=BiographicalTimelineDetailLevel.LOW,
        trauma_level="none",
        age_range=(20, 25),
    )
    assert m.is_forgotten is False  # Only 4 months, needs >= 6


def test_invalid_trauma_level_raises():
    with pytest.raises(ValidationError):
        BiographicalPeriod(
            start_date=datetime.date(1997, 1, 1),
            end_date=datetime.date(1997, 7, 31),
            description="test",
            trauma_level="extreme",
            age_range=(20, 25),
        )


def test_valid_trauma_levels():
    valid_levels = ["none", "low", "medium", "high"]
    for level in valid_levels:
        m = BiographicalPeriod(
            start_date=datetime.date(1997, 1, 1),
            end_date=datetime.date(1997, 7, 31),
            description="test",
            trauma_level=level,
            age_range=(20, 25),
        )
        assert m.trauma_level == level


def test_field_descriptions():
    fields = getattr(BiographicalPeriod, "model_fields", None)
    assert fields is not None
