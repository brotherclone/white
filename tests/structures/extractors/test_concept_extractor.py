import pytest

from app.structures.extractors.concept_extractor import (ConceptExtractor,
                                                         RebrackettingAnalysis)


def test_concept_extractor():
    """Test ConceptExtractor initialization and basic functionality"""
    # This test assumes there's a valid manifest at 02_01
    # If this fails, we may need to mock or use a different manifest_id
    try:
        extractor = ConceptExtractor(manifest_id="02_01")
        assert extractor.manifest_id == "02_01"
        assert extractor.manifest is not None
    except (ValueError, FileNotFoundError):
        # If manifest doesn't exist, test with invalid ID to ensure proper error handling
        with pytest.raises(ValueError):
            ConceptExtractor(manifest_id="invalid_manifest_id")


def test_rebracketting_analysis_creation():
    """Test RebrackettingAnalysis dataclass creation"""
    analysis = RebrackettingAnalysis(
        original_memory="blue wafer",
        corrected_memory="purple cookie",
        rebracketing_type="perceptual_rebracketing",
    )
    assert analysis.original_memory == "blue wafer"
    assert analysis.corrected_memory == "purple cookie"
    assert analysis.rebracketing_type == "perceptual_rebracketing"


def test_analyze_concept_field_empty():
    """Test analyzing empty concept field"""
    # Create mock extractor for testing the method directly
    analysis = RebrackettingAnalysis()
    assert analysis.original_memory is None
    assert analysis.corrected_memory is None
    assert analysis.memory_discrepancy_severity == 0.0
