import pytest
from pydantic import ValidationError

from app.structures.manifests.manifest_sounds_like import ManifestSoundsLike


def test_manifest_sounds_like_required_fields():
    s = ManifestSoundsLike(discogs_id=123, name="Test Artist")
    assert s.discogs_id == 123
    assert s.name == "Test Artist"


def test_manifest_sounds_like_missing_required():
    with pytest.raises(ValidationError):
        ManifestSoundsLike(name="Test Artist")
    with pytest.raises(ValidationError):
        ManifestSoundsLike(discogs_id=123)
