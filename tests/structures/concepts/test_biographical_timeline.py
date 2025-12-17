import pytest
from pydantic import ValidationError
import datetime

from app.structures.concepts.biographical_timeline import BiographicalTimeline
from app.structures.concepts.biographical_period import BiographicalPeriod


def test_create_valid_model():
    period1 = BiographicalPeriod(
        start_date=datetime.date(1997, 1, 1),
        end_date=datetime.date(1997, 12, 31),
        description="First period",
        age_range=(20, 25),
    )
    period2 = BiographicalPeriod(
        start_date=datetime.date(1998, 1, 1),
        end_date=datetime.date(1998, 12, 31),
        description="Second period",
        age_range=(25, 30),
    )

    m = BiographicalTimeline(periods=[period1, period2])
    assert isinstance(m, BiographicalTimeline)
    assert len(m.periods) == 2
    assert m.high_detail_periods == []
    assert m.low_detail_periods == []
    assert m.forgotten_periods == []
    assert m.model_dump(exclude_none=True, exclude_unset=True)


def test_with_optional_fields():
    period = BiographicalPeriod(
        start_date=datetime.date(1997, 1, 1),
        end_date=datetime.date(1997, 12, 31),
        description="test",
        age_range=(20, 25),
    )

    m = BiographicalTimeline(
        periods=[period],
        total_span_years=1,
        high_detail_periods=[period],
        low_detail_periods=[],
        forgotten_periods=[],
    )
    assert m.total_span_years == 1
    assert len(m.high_detail_periods) == 1


def test_missing_required_field_raises():
    with pytest.raises(ValidationError):
        BiographicalTimeline()


def test_wrong_type_raises():
    with pytest.raises(ValidationError):
        BiographicalTimeline(periods="not a list")


def test_get_surrounding_periods_middle():
    period1 = BiographicalPeriod(
        start_date=datetime.date(1996, 1, 1),
        end_date=datetime.date(1996, 12, 31),
        description="First period",
        age_range=(20, 25),
    )
    period2 = BiographicalPeriod(
        start_date=datetime.date(1997, 1, 1),
        end_date=datetime.date(1997, 12, 31),
        description="Second period",
        age_range=(25, 30),
    )
    period3 = BiographicalPeriod(
        start_date=datetime.date(1998, 1, 1),
        end_date=datetime.date(1998, 12, 31),
        description="Third period",
        age_range=(30, 35),
    )

    timeline = BiographicalTimeline(periods=[period1, period2, period3])
    surrounding = timeline.get_surrounding_periods(period2)

    assert surrounding["preceding"] == period1
    assert surrounding["following"] == period3


def test_get_surrounding_periods_first():
    period1 = BiographicalPeriod(
        start_date=datetime.date(1996, 1, 1),
        end_date=datetime.date(1996, 12, 31),
        description="First period",
        age_range=(20, 25),
    )
    period2 = BiographicalPeriod(
        start_date=datetime.date(1997, 1, 1),
        end_date=datetime.date(1997, 12, 31),
        description="Second period",
        age_range=(25, 30),
    )

    timeline = BiographicalTimeline(periods=[period1, period2])
    surrounding = timeline.get_surrounding_periods(period1)

    assert surrounding["preceding"] is None
    assert surrounding["following"] == period2


def test_get_surrounding_periods_last():
    period1 = BiographicalPeriod(
        start_date=datetime.date(1996, 1, 1),
        end_date=datetime.date(1996, 12, 31),
        description="First period",
        age_range=(20, 25),
    )
    period2 = BiographicalPeriod(
        start_date=datetime.date(1997, 1, 1),
        end_date=datetime.date(1997, 12, 31),
        description="Second period",
        age_range=(25, 30),
    )

    timeline = BiographicalTimeline(periods=[period1, period2])
    surrounding = timeline.get_surrounding_periods(period2)

    assert surrounding["preceding"] == period1
    assert surrounding["following"] is None


def test_get_surrounding_periods_single_period():
    period = BiographicalPeriod(
        start_date=datetime.date(1997, 1, 1),
        end_date=datetime.date(1997, 12, 31),
        description="Only period",
        age_range=(20, 25),
    )

    timeline = BiographicalTimeline(periods=[period])
    surrounding = timeline.get_surrounding_periods(period)

    assert surrounding["preceding"] is None
    assert surrounding["following"] is None


def test_filter_by_age_range():
    period1 = BiographicalPeriod(
        start_date=datetime.date(1996, 1, 1),
        end_date=datetime.date(1996, 12, 31),
        description="First period",
        age_range=(18, 19),
    )

    period2 = BiographicalPeriod(
        start_date=datetime.date(1997, 1, 1),
        end_date=datetime.date(1997, 12, 31),
        description="Second period",
        age_range=(22, 23),
    )

    period3 = BiographicalPeriod(
        start_date=datetime.date(1998, 1, 1),
        end_date=datetime.date(1998, 12, 31),
        description="Third period",
        age_range=(25, 26),
    )

    timeline = BiographicalTimeline(periods=[period1, period2, period3])

    # Filter for ages 20-24
    filtered = timeline.filter_by_age_range(20, 24)
    assert len(filtered) == 1
    assert filtered[0] == period2


def test_filter_by_age_range_multiple_matches():
    period1 = BiographicalPeriod(
        start_date=datetime.date(1996, 1, 1),
        end_date=datetime.date(1996, 12, 31),
        description="First period",
        age_range=(22, 23),
    )

    period2 = BiographicalPeriod(
        start_date=datetime.date(1997, 1, 1),
        end_date=datetime.date(1997, 12, 31),
        description="Second period",
        age_range=(23, 24),
    )

    period3 = BiographicalPeriod(
        start_date=datetime.date(1998, 1, 1),
        end_date=datetime.date(1998, 12, 31),
        description="Third period",
        age_range=(30, 31),
    )

    timeline = BiographicalTimeline(periods=[period1, period2, period3])

    # Filter for ages 20-25
    filtered = timeline.filter_by_age_range(20, 25)
    assert len(filtered) == 2
    assert period1 in filtered
    assert period2 in filtered


def test_filter_by_age_range_no_matches():
    period = BiographicalPeriod(
        start_date=datetime.date(1997, 1, 1),
        end_date=datetime.date(1997, 12, 31),
        description="Period",
        age_range=(20, 25),
    )
    period.age_range = (18, 19)

    timeline = BiographicalTimeline(periods=[period])

    # Filter for ages that don't match
    filtered = timeline.filter_by_age_range(30, 40)
    assert len(filtered) == 0


def test_field_descriptions():
    fields = getattr(BiographicalTimeline, "model_fields", None)
    assert fields is not None
