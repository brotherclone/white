from app.structures.extractors.midi_extractor import MidiExtractor


def test_midi_extractor():
    """Test MidiExtractor initialization"""
    try:
        extractor = MidiExtractor(manifest_id="02_01")
        assert extractor.manifest_id == "02_01"
    except (ValueError, FileNotFoundError):
        # If manifest doesn't exist, test with invalid ID
        extractor = MidiExtractor(manifest_id="nonexistent_manifest")
        assert extractor.manifest_id == "nonexistent_manifest"
        assert extractor.manifest is None


def test_midi_extractor_extract_segment_features_invalid_file():
    """Test extracting MIDI features from invalid file returns defaults"""
    extractor = MidiExtractor(manifest_id="test")
    features = extractor.extract_segment_features(
        "/nonexistent/file.mid", start_time=0.0, end_time=1.0
    )
    assert isinstance(features, dict)
    assert features["note_density"] == 0.0
    assert features["pitch_variety"] == 0.0
    # rhythmic_regularity is 0 when there's an error (from _empty_midi_features)
    assert features["rhythmic_regularity"] == 0
    assert features["avg_polyphony"] == 0.0


def test_midi_extractor_load_midi_segment_error():
    """Test load_midi_segment with invalid path returns empty features dict"""
    extractor = MidiExtractor(manifest_id="test")
    result = extractor.load_midi_segment(
        "/invalid/path.mid", start_time=0.0, end_time=1.0
    )
    # Should return dict with empty features, not None
    assert isinstance(result, dict)
    assert result["event_summary"]["total_events"] == 0
    assert result["rhythmic_regularity"] == 0
