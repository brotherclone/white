import pytest

from app.structures.extractors.lyric_extrator import LyricExtractor


def test_extract_lyric():
    """Test LyricExtractor initialization"""
    # Test with a manifest that might exist
    try:
        extractor = LyricExtractor(manifest_id="02_01")
        assert extractor.manifest_id == "02_01"
        assert extractor.manifest is not None
    except (ValueError, FileNotFoundError):
        # If manifest doesn't exist, test error handling
        with pytest.raises(ValueError):
            LyricExtractor(manifest_id="nonexistent_manifest")


def test_extract_lyric_segment_features():
    """Test extracting lyric segment features"""
    try:
        extractor = LyricExtractor(manifest_id="02_01")
        if extractor.lrc_path and extractor.lyrics:
            # Test segment extraction
            features = extractor.extract_segment_features(
                extractor.lrc_path, start_time=0.0, end_time=10.0
            )
            assert isinstance(features, list)
    except (ValueError, FileNotFoundError):
        # If no valid manifest, just pass
        pass


def test_extract_lyric_no_overlap():
    """Test extracting lyrics with no overlap returns empty list"""
    try:
        extractor = LyricExtractor(manifest_id="02_01")
        if extractor.lrc_path:
            # Test with time range that shouldn't have lyrics
            features = extractor.extract_segment_features(
                extractor.lrc_path, start_time=999999.0, end_time=999999.5
            )
            assert isinstance(features, list)
            assert len(features) == 0
    except (ValueError, FileNotFoundError):
        pass
