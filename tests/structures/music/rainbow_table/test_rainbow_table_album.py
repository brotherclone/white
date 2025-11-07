from datetime import datetime

import pytest
from pydantic import ValidationError

from app.structures.music.rainbow_table.rainbow_table_album import RainbowTableAlbum


def test_rainbow_table_album():
    """Test RainbowTableAlbum creation with required fields"""
    album = RainbowTableAlbum(title="Test Album", release_date=datetime(2025, 1, 1))
    assert album.title == "Test Album"
    assert album.release_date == datetime(2025, 1, 1)
    assert album.ef_id is None
    assert album.rainbow_mnemonic_character_value is None


def test_rainbow_table_album_with_optional_fields():
    """Test RainbowTableAlbum with all fields"""
    album = RainbowTableAlbum(
        title="Complete Album",
        release_date=datetime(2024, 6, 15),
        ef_id="EF-12345",
        rainbow_mnemonic_character_value="R",
    )
    assert album.title == "Complete Album"
    assert album.ef_id == "EF-12345"
    assert album.rainbow_mnemonic_character_value == "R"


def test_rainbow_table_album_missing_required():
    """Test that required fields are enforced"""
    with pytest.raises(ValidationError):
        RainbowTableAlbum(title="Missing Date")

    with pytest.raises(ValidationError):
        RainbowTableAlbum(release_date=datetime.now())


def test_rainbow_table_album_ef_id_types():
    """Test that ef_id accepts string or int"""
    album1 = RainbowTableAlbum(title="Album 1", release_date=datetime.now(), ef_id=123)
    assert album1.ef_id == 123

    album2 = RainbowTableAlbum(
        title="Album 2", release_date=datetime.now(), ef_id="ABC-123"
    )
    assert album2.ef_id == "ABC-123"
