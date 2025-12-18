from app.reference.biographical.places_frequented import (
    NEW_JERSEY_LOCATIONS,
    NEW_YORK_LOCATIONS,
    MARYLAND_LOCATIONS,
    DMV_LOCATIONS,
    UK_LOCATIONS,
    IRELAND_LOCATIONS,
    DANISH_LOCATIONS,
    EUROPE_LOCATIONS,
    US_LOCATIONS,
    ALL_LOCATIONS,
)


def test_new_jersey_locations_not_empty():
    assert len(NEW_JERSEY_LOCATIONS) > 0
    assert "Wantage" in NEW_JERSEY_LOCATIONS


def test_new_york_locations_not_empty():
    assert len(NEW_YORK_LOCATIONS) > 0
    assert "Brooklyn" in NEW_YORK_LOCATIONS


def test_maryland_locations_not_empty():
    assert len(MARYLAND_LOCATIONS) > 0
    assert "Baltimore" in MARYLAND_LOCATIONS


def test_dmv_locations_includes_maryland():
    assert len(DMV_LOCATIONS) > len(MARYLAND_LOCATIONS)
    assert "Baltimore" in DMV_LOCATIONS
    assert "Washington DC" in DMV_LOCATIONS


def test_uk_locations_not_empty():
    assert len(UK_LOCATIONS) > 0
    assert "London" in UK_LOCATIONS


def test_ireland_locations_not_empty():
    assert len(IRELAND_LOCATIONS) > 0
    assert "Dublin" in IRELAND_LOCATIONS


def test_danish_locations_not_empty():
    assert len(DANISH_LOCATIONS) > 0
    assert "Copenhagen" in DANISH_LOCATIONS


def test_europe_locations_combines_all_european():
    assert len(EUROPE_LOCATIONS) == len(UK_LOCATIONS) + len(IRELAND_LOCATIONS) + len(
        DANISH_LOCATIONS
    )
    assert "London" in EUROPE_LOCATIONS
    assert "Dublin" in EUROPE_LOCATIONS
    assert "Copenhagen" in EUROPE_LOCATIONS


def test_us_locations_combines_all_us():
    assert len(US_LOCATIONS) == len(NEW_JERSEY_LOCATIONS) + len(
        NEW_YORK_LOCATIONS
    ) + len(DMV_LOCATIONS)
    assert "Wantage" in US_LOCATIONS
    assert "Brooklyn" in US_LOCATIONS


def test_all_locations_combines_everything():
    assert len(ALL_LOCATIONS) == len(EUROPE_LOCATIONS) + len(US_LOCATIONS)
    assert "London" in ALL_LOCATIONS
    assert "Wantage" in ALL_LOCATIONS
