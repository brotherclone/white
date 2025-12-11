"""
Concept Extractor - Updated to use clean structure.

Returns ConceptExtractionResult with no redundancy.
"""

from typing import Optional
from pydantic import BaseModel, Field

from app.extractors.manifest_extractor.scoring_functions import (
    score_rebracketing_intensity,
    score_temporal_complexity,
    score_ontological_uncertainty,
    score_memory_discrepancy,
    score_boundary_fluidity,
    score_rebracketing_coverage,
    check_has_rebracketing_markers,
    calculate_basic_text_features,
)
from app.extractors.manifest_extractor.rebracketing_type_classifier import (
    classify_by_domain,
)
from app.structures.concepts.methodology_feature import MethodologyFeature
from app.structures.concepts.rainbow_table_color import get_rainbow_table_color
from app.structures.concepts.rebracketing_analysis import RebracketingAnalysis
from app.extractors.manifest_extractor.concept_extraction_result import (
    ConceptExtractionResult,
)


class ConceptExtractor(BaseModel):
    """
    Extracts and analyzes rebracketing concepts from track metadata.

    Returns ConceptExtractionResult with:
    - MethodologyFeature (pure features, no identifiers)
    - RebracketingAnalysis (results with track_id)
    - No redundancy

    Usage:
        extractor = ConceptExtractor(
            track_id="1_5",
            concept_text="The injury was different...",
            lyric_text="Optional lyrics"
        )

        result = extractor.extract()
        training_dict = result.to_training_dict()
    """

    track_id: str = Field(description="Track identifier (album_track format)")
    concept_text: str = Field(description="The concept text to analyze")
    rainbow_color_mnemonic: Optional[str] = Field(default=None)
    lyric_text: Optional[str] = Field(default=None)
    track_duration: Optional[float] = Field(default=None)

    # Internal state
    _original_concept: Optional[str] = None
    _full_text: Optional[str] = None
    _album_sequence: Optional[int] = None
    _track_sequence: Optional[int] = None

    def __init__(self, **data):
        super().__init__(**data)

        if not self.track_id:
            raise ValueError("Track identifier cannot be None")
        if not self.concept_text:
            raise ValueError("Concept text cannot be None")

        self._parse_track_id()
        self._original_concept = self.concept_text
        self._full_text = self._prepare_text()

    def _parse_track_id(self):
        """Extract album and track sequence from track_id"""
        try:
            parts = self.track_id.split("_")
            self._album_sequence = int(parts[0])
            self._track_sequence = int(parts[1])
        except (IndexError, ValueError):
            pass

    def _prepare_text(self) -> str:
        """Combine concept and lyrics, normalize for analysis"""
        text = self.concept_text
        if self.lyric_text:
            text = f"{text} {self.lyric_text}"
        return text.strip().lower()

    def extract(self) -> ConceptExtractionResult:
        """
        Main extraction method - returns complete result.

        Returns:
            ConceptExtractionResult with features and analysis
        """
        features = self._extract_methodology_features()
        analysis = self._extract_rebracketing_analysis()

        return ConceptExtractionResult(
            track_id=self.track_id,
            methodology_features=features,
            rebracketing_analysis=analysis,
        )

    def _extract_methodology_features(self) -> MethodologyFeature:
        """Generate methodology feature scores"""
        # Basic text features
        basic_features = calculate_basic_text_features(self._original_concept)

        # Methodology scores (on full normalized text)
        return MethodologyFeature(
            # Text statistics
            concept_length=basic_features["concept_length"],
            word_count=basic_features["word_count"],
            sentence_count=basic_features["sentence_count"],
            avg_word_length=basic_features["avg_word_length"],
            question_marks=basic_features["question_marks"],
            exclamation_marks=basic_features["exclamation_marks"],
            # Methodology scores
            uncertainty_level=score_boundary_fluidity(self._full_text),
            narrative_complexity=score_temporal_complexity(self._full_text),
            discrepancy_intensity=score_memory_discrepancy(self._full_text),
            has_rebracketing_markers=check_has_rebracketing_markers(self._full_text),
            # Context features
            track_duration=self.track_duration,
            track_position=self._track_sequence,
            album_sequence=self._album_sequence,
        )

    def _extract_rebracketing_analysis(self) -> RebracketingAnalysis:
        """Generate full rebracketing analysis"""
        return RebracketingAnalysis(
            track_id=self.track_id,
            rebracketing_type=classify_by_domain(self._full_text),
            rebracketing_intensity=score_rebracketing_intensity(self._full_text),
            rebracketing_coverage=score_rebracketing_coverage(self._full_text),
            ontological_uncertainty=score_ontological_uncertainty(self._full_text),
            memory_discrepancy_severity=score_memory_discrepancy(self._full_text),
            temporal_complexity_score=score_temporal_complexity(self._full_text),
            boundary_fluidity_score=score_boundary_fluidity(self._full_text),
            original_memory=self._original_concept,
            concept_text_analyzed=self._full_text,
            rainbow_color=get_rainbow_table_color(self.rainbow_color_mnemonic),
        )

    # Convenience methods for backward compatibility
    def get_methodology_features(self) -> MethodologyFeature:
        """Backward compatible method"""
        return self._extract_methodology_features()

    def get_rebracketing_analysis(self) -> RebracketingAnalysis:
        """Backward compatible method"""
        return self._extract_rebracketing_analysis()
