import os
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Dict, List

from app.structures.enums.temporal_relatioship import TemporalRelationship
from app.util.manifest_loader import load_manifest
from app.util.manifest_validator import validate_yaml_file
from app.util.lrc_utils import load_lrc

class ManifestExtractor:
    """Main orchestrator class for extracting data from manifests using specialized extractors"""

    def __init__(self, mani_id: str):
        load_dotenv()
        self.manifest_id = mani_id
        self.manifest_path = os.path.join(
            os.environ['MANIFEST_PATH'],
            mani_id,
            f"{mani_id}.yml"
        )

        if not os.path.exists(self.manifest_path):
            raise ValueError(f"Manifest file not found: {self.manifest_path}")

        # Validate the manifest
        all_valid, errors = validate_yaml_file(self.manifest_path)
        if all_valid:
            print("YAML file valid.")
        else:
            print("YAML file has errors:")
            for error in errors:
                print(f" - {error}")

        # Load the manifest
        self.manifest = load_manifest(self.manifest_path)
        if self.manifest is None:
            raise ValueError("Manifest could not be loaded.")

        # Initialize component extractors (will be imported when needed to avoid circular imports)
        self._concept_extractor = None
        self._lyric_extractor = None
        self._audio_extractor = None
        self._midi_extractor = None

    @property
    def concept_extractor(self):
        """Lazy load concept extractor to avoid circular imports"""
        if self._concept_extractor is None:
            from app.structures.extractors.concept_extractor import ConceptExtractor
            self._concept_extractor = ConceptExtractor(manifest_id=self.manifest_id)
        return self._concept_extractor

    @property
    def lyric_extractor(self):
        """Lazy load lyric extractor to avoid circular imports"""
        if self._lyric_extractor is None:
            from app.structures.extractors.lyric_extrator import LyricExtractor
            self._lyric_extractor = LyricExtractor(manifest_id=self.manifest_id)
        return self._lyric_extractor

    @property
    def audio_extractor(self):
        """Lazy load audio extractor to avoid circular imports"""
        if self._audio_extractor is None:
            from app.structures.extractors.audio_extractor import AudioExtractor
            self._audio_extractor = AudioExtractor(manifest_id=self.manifest_id)
        return self._audio_extractor

    @property
    def midi_extractor(self):
        """Lazy load MIDI extractor to avoid circular imports"""
        if self._midi_extractor is None:
            from app.structures.extractors.midi_extractor import MidiExtractor
            self._midi_extractor = MidiExtractor(manifest_id=self.manifest_id)
        return self._midi_extractor

    def parse_yaml_time(self, time_str: str) -> float:
        """Parse YAML timestamp like '[00:28.086]' to seconds"""
        return self.parse_lrc_time(time_str)


    def parse_lrc_time(self, time_str: str) -> float:
        """Parse LRC timestamp like '[00:28.086]' to seconds"""
        try:
            time_str = time_str.strip().strip('[]')
            minutes, seconds = time_str.split(':')
            total_seconds = int(minutes) * 60 + float(seconds)
            return total_seconds
        except Exception as e:
            raise ValueError(f"Invalid LRC time format: {time_str}") from e

    @staticmethod
    def determine_temporal_relationship(content_start: float, content_end: float,
                                        segment_start: float, segment_end: float) -> str:
        """Determine how content relates to segment boundaries"""
        if content_start < segment_start and content_end > segment_end:
            return TemporalRelationship.ACROSS
        elif content_start < segment_start:
            return TemporalRelationship.BLEED_IN
        elif content_end > segment_end:
            return TemporalRelationship.BLEED_OUT
        else:
            return TemporalRelationship.CONTAINED


    @staticmethod
    def _calculate_boundary_fluidity_score(segment: Dict[str, Any]) -> float:
        """Calculate how much this segment exhibits boundary fluidity across modalities"""
        score = 0.0

        # Lyrical boundary fluidity
        lyric_bleeding = len([l for l in segment['lyrical_content']
                              if l['temporal_relationship'] not in [TemporalRelationship.CONTAINED, TemporalRelationship.MATCH]])
        score += lyric_bleeding * 0.3

        # Audio boundary fluidity (based on attack/decay profiles)
        if 'audio_features' in segment:
            audio = segment['audio_features']
            # Fast attack indicates abrupt boundary
            if audio.get('attack_time', 0) < 0.1:
                score += 0.2
            # Complex decay profile indicates temporal spreading
            if len(audio.get('decay_profile', [])) > 0:
                decay_var = np.var(audio['decay_profile'])
                score += min(decay_var * 0.1, 0.3)

        # MIDI boundary fluidity (based on note overlaps and timing)
        if 'midi_features' in segment:
            midi = segment['midi_features']
            # Irregular timing suggests boundary crossing
            if midi.get('rhythmic_regularity', 1) < 0.5:
                score += 0.2
            # High polyphony suggests complex overlapping
            if midi.get('avg_polyphony', 0) > 2:
                score += 0.2

        return score

    def load_audio_segment(self, audio_path: str, start_time: float, end_time: float) -> Dict[str, Any]:
        """Load audio features for a specific time segment"""
        try:
            return self.audio_extractor.extract_segment_features(audio_path, start_time, end_time)
        except Exception as e:
            print(f"Warning: Could not load audio features: {e}")
            return {
                'rms_energy': 0.0,
                'spectral_centroid': 0.0,
                'attack_time': 0.0,
                'decay_profile': []
            }

    def load_midi_segment(self, midi_path: str, start_time: float, end_time: float) -> Dict[str, Any]:
        """Load MIDI features for a specific time segment"""
        try:
            return self.midi_extractor.extract_segment_features(midi_path, start_time, end_time)
        except Exception as e:
            print(f"Warning: Could not load MIDI features: {e}")
            return {
                'note_density': 0.0,
                'pitch_variety': 0.0,
                'rhythmic_regularity': 1.0,
                'avg_polyphony': 0.0
            }

    def generate_multimodal_segments(self, mms_yaml_path: str, mms_lrc_path: str = None,
                                     mms_audio_path: str = None, mms_midi_path: str = None) -> pd.DataFrame:
        """Generate complete multimodal training dataset"""
        print(f"=== MULTIMODAL REBRACKETING TRAINING DATA GENERATION ===")
        print(f"YAML: {mms_yaml_path}")
        print(f"LRC:  {mms_lrc_path}")
        print(f"Audio: {mms_audio_path}")
        print(f"MIDI: {mms_midi_path}")

        manifest = load_manifest(mms_yaml_path)
        print(f"✅ Loaded manifest: {getattr(manifest,'title', 'Unknown')}")

        # Load lyrics using LyricExtractor if provided
        intersecting_lyrics_by_section = {}
        if mms_lrc_path and Path(mms_lrc_path).exists():
            print(f"✅ Using LyricExtractor for {mms_lrc_path}")

        all_segments = []

        # Generate section-based segments with multimodal data
        for section in manifest.structure:
            section_start = self.parse_yaml_time(section.start_time)
            section_end = self.parse_yaml_time(section.end_time)

            # Extract intersecting lyrics using LyricExtractor
            intersecting_lyrics = []
            if mms_lrc_path and Path(mms_lrc_path).exists():
                try:
                    intersecting_lyrics = self.lyric_extractor.extract_segment_features(
                        mms_lrc_path, section_start, section_end
                    )
                except Exception as e:
                    print(f"Warning: Could not extract lyric features: {e}")
                    intersecting_lyrics = []

            # Process lyrics to add temporal relationship info
            processed_lyrics = []
            for lyric in intersecting_lyrics:
                lyric_start = lyric['start_time']
                lyric_end = lyric['end_time']

                temporal_rel = self.determine_temporal_relationship(
                    lyric_start, lyric_end, section_start, section_end
                )

                overlap_start = max(lyric_start, section_start)
                overlap_end = min(lyric_end, section_end)
                overlap_duration = max(0, overlap_end - overlap_start)
                lyric_duration = lyric_end - lyric_start
                confidence = overlap_duration / lyric_duration if lyric_duration > 0 else 0

                processed_lyrics.append({
                    'text': lyric['text'],
                    'start_time': lyric_start,
                    'end_time': lyric_end,
                    'temporal_relationship': temporal_rel,
                    'confidence': confidence,
                    'overlap_seconds': overlap_duration
                })

            # ToDo: new class for Segment
            segment = {
                'manifest_id': manifest.manifest_id,
                'segment_type': 'section',
                'segment_id': f"{manifest['manifest_id']}_{section['section_name'].lower().replace(' ', '_')}",
                'canonical_start': section_start,
                'canonical_end': section_end,
                'duration': section_end - section_start,
                'section_name': section.section_name,
                'section_description': section.description,
                'lyrical_content': processed_lyrics,

                # Musical context
                'bpm': getattr(manifest, 'bpm', None),
                'time_signature': getattr(manifest, 'tempo', None),
                'key': getattr(manifest, 'key', None),
                'rainbow_color': getattr(manifest, 'rainbow_color', None),

                # Metadata
                'title': getattr(manifest, 'title', None),
                'mood_tags': getattr(manifest, 'mood', []),
                'concept': getattr(manifest, 'concept', ''),
            }

            # Add audio features if audio file provided
            if mms_audio_path and Path(mms_audio_path).exists():
                segment['audio_features'] = self.load_audio_segment(mms_audio_path, section_start, section_end)
                print(f"✅ Added audio features for {section.section_name}")

            # Add MIDI features if MIDI file provided
            if mms_midi_path and Path(mms_midi_path).exists():
                segment['midi_features'] = self.load_midi_segment(mms_midi_path, section_start, section_end)
                print(f"✅ Added MIDI features for {section.section_name}")

            # Calculate boundary fluidity metrics across all modalities
            segment['rebracketing_score'] = self._calculate_boundary_fluidity_score(segment)

            all_segments.append(segment)

        df = pd.DataFrame(all_segments)

        # Add computed multimodal metrics
        df['lyric_count'] = df['lyrical_content'].apply(len)
        df['has_temporal_bleeding'] = df['lyrical_content'].apply(
            lambda x: any(l['temporal_relationship'] not in ['contained', 'exact_match'] for l in x)
        )

        if mms_audio_path:
            df['audio_energy'] = df['audio_features'].apply(lambda x: x.get('rms_energy', 0))
            df['spectral_complexity'] = df['audio_features'].apply(lambda x: x.get('spectral_centroid', 0))

        if mms_midi_path:
            df['midi_density'] = df['midi_features'].apply(lambda x: x.get('note_density', 0))
            df['pitch_complexity'] = df['midi_features'].apply(lambda x: x.get('pitch_variety', 0))

        print(f"\n=== MULTIMODAL ANALYSIS ===")
        print(f"Total segments: {len(df)}")
        print(f"Segments with lyrics: {(df['lyric_count'] > 0).sum()}")
        print(f"Segments with temporal bleeding: {df['has_temporal_bleeding'].sum()}")

        if mms_audio_path:
            print(f"Average audio energy: {df['audio_energy'].mean():.4f}")
        if mms_midi_path:
            print(f"Average MIDI note density: {df['midi_density'].mean():.2f} notes/second")

        return df


    def generate_enhanced_multimodal_segments(self, mm_yaml_path: str, mm_lrc_path: str = None,
                                              mm_audio_path: str = None, mm_midi_path: str = None) -> pd.DataFrame:
        """Generate multimodal training dataset with rebracketing methodology analysis"""
        print(f"=== ENHANCED MULTIMODAL REBRACKETING TRAINING DATA GENERATION ===")
        print(f"YAML: {mm_yaml_path}")
        print(f"LRC:  {mm_lrc_path}")
        print(f"Audio: {mm_audio_path}")
        print(f"MIDI: {mm_midi_path}")

        manifest = load_manifest(mm_yaml_path)
        print(f"✅ Loaded manifest: {getattr(manifest, 'title', 'Unknown')}")
        # Avoid circular import
        from app.structures.extractors.concept_extractor import ConceptExtractor
        # Initialize concept extractor for rebracketing analysis
        concept_extractor = ConceptExtractor(manifest_id=self.manifest_id)
        rebracketing_features = concept_extractor.generate_rebracketing_training_features()
        concept_analysis = concept_extractor.concept_analysis

        print(f"✅ Analyzed concept field - Rebracketing Type: {concept_analysis.rebracketing_type}")
        print(f"   Memory Discrepancy: {concept_analysis.original_memory} → {concept_analysis.corrected_memory}")

        # Load lyrics if provided
        lyrics = []
        if mm_lrc_path and Path(mm_lrc_path).exists():
            lyrics = load_lrc(mm_lrc_path)
            print(f"✅ Loaded {len(lyrics)} lyrics")

        all_segments = []

        # Generate section-based segments with enhanced rebracketing analysis
        for section in manifest['structure']:
            section_start = self.parse_yaml_time(section['start_time'])
            section_end = self.parse_yaml_time(section['end_time'])

            # Find intersecting lyrics with enhanced temporal analysis
            intersecting_lyrics = []
            if lyrics:
                for lyric in lyrics:
                    lyric_start = lyric['start_time']
                    lyric_end = lyric['end_time']

                    if not (lyric_end <= section_start or lyric_start >= section_end):
                        temporal_rel = self.determine_temporal_relationship(
                            lyric_start, lyric_end, section_start, section_end
                        )

                        overlap_start = max(lyric_start, section_start)
                        overlap_end = min(lyric_end, section_end)
                        overlap_duration = max(0, overlap_end - overlap_start)
                        lyric_duration = lyric_end - lyric_start
                        confidence = overlap_duration / lyric_duration if lyric_duration > 0 else 0

                        # Enhanced lyric analysis for rebracketing patterns
                        lyric_data = {
                            'text': lyric['text'],
                            'start_time': lyric_start,
                            'end_time': lyric_end,
                            'temporal_relationship': temporal_rel,
                            'confidence': confidence,
                            'overlap_seconds': overlap_duration,

                            # NEW: Rebracketing-specific lyrical analysis
                            'contains_memory_reference': self._check_memory_references(lyric['text']),
                            'contains_temporal_markers': self._check_temporal_markers(lyric['text']),
                            'contains_rebracketing_language': self._check_rebracketing_language(lyric['text']),
                            'lyrical_boundary_fluidity': self._calculate_lyrical_boundary_fluidity(temporal_rel, confidence)
                        }
                        intersecting_lyrics.append(lyric_data)

            # Enhanced segment with rebracketing methodology
            segment = {
                'manifest_id': manifest['manifest_id'],
                'segment_type': 'section',
                'segment_id': f"{manifest['manifest_id']}_{section['section_name'].lower().replace(' ', '_')}",
                'canonical_start': section_start,
                'canonical_end': section_end,
                'duration': section_end - section_start,
                'section_name': section['section_name'],
                'section_description': section['description'],
                'lyrical_content': intersecting_lyrics,

                # Musical context
                'bpm': getattr(manifest,'bpm'),
                'time_signature':getattr(manifest,'tempo'),
                'key': getattr(manifest,'key'),
                'rainbow_color': getattr(manifest,'rainbow_color'),

                # Enhanced metadata with rebracketing analysis
                'title':getattr(manifest,'title'),
                'mood_tags':getattr(manifest,'mood', []),
                'concept': getattr(manifest,'concept', ''),

                # NEW: Rebracketing methodology features
                'rebracketing_features': rebracketing_features,
                'concept_analysis': concept_analysis.__dict__ if concept_analysis else {},
                'ontological_category': concept_analysis.ontological_category if concept_analysis else None,
                'memory_discrepancy_severity': concept_analysis.memory_discrepancy_severity if concept_analysis else 0.0,
                'temporal_complexity': rebracketing_features.get('temporal_rebracketing_complexity', 0.0),

                # Section-specific rebracketing analysis
                'section_rebracketing_score': self._calculate_section_rebracketing_score(
                    section, intersecting_lyrics, concept_analysis
                ),
                'boundary_crossing_indicators': self._identify_boundary_crossing_indicators(
                    section, intersecting_lyrics
                )

            }

            # Add audio features with rebracketing analysis
            if mm_audio_path and Path(mm_audio_path).exists():
                audio_features = self.load_audio_segment(mm_audio_path, section_start, section_end)
                segment['audio_features'] = audio_features

                # NEW: Audio-based rebracketing metrics
                segment['audio_rebracketing_metrics'] = {
                    'spectral_instability': self._calculate_spectral_instability(audio_features),
                    'temporal_discontinuity': self._calculate_temporal_discontinuity(audio_features),
                    'genre_shift_indicators': self._detect_genre_shift_indicators(audio_features, section)
                }
                print(f"✅ Added enhanced audio features for {section['section_name']}")

            # Add MIDI features with rebracketing analysis
            if mm_midi_path and Path(mm_midi_path).exists():
                midi_features = self.load_midi_segment(mm_midi_path, section_start, section_end)
                segment['midi_features'] = midi_features

                # NEW: MIDI-based rebracketing metrics
                segment['midi_rebracketing_metrics'] = {
                    'harmonic_discontinuity': self._calculate_harmonic_discontinuity(midi_features),
                    'rhythmic_rebracketing': self._calculate_rhythmic_rebracketing(midi_features),
                    'instrumental_genre_mixing': self._detect_instrumental_genre_mixing(midi_features)
                }
                print(f"✅ Added enhanced MIDI features for {section['section_name']}")

            # Calculate comprehensive boundary fluidity across all modalities
            segment['comprehensive_rebracketing_score'] = self._calculate_comprehensive_rebracketing_score(segment)

            all_segments.append(segment)

        df = pd.DataFrame(all_segments)

        # Enhanced computed metrics for rebracketing training
        df['lyric_count'] = df['lyrical_content'].apply(len)
        df['has_temporal_bleeding'] = df['lyrical_content'].apply(
            lambda x: any(l['temporal_relationship'] not in ['contained', 'exact_match'] for l in x)
        )

        # NEW: Rebracketing-specific computed columns
        df['follows_orange_methodology'] = df['rebracketing_features'].apply(
            lambda x: x.get('follows_orange_pattern', False)
        )
        df['memory_boundary_crossing'] = df['rebracketing_features'].apply(
            lambda x: x.get('memory_boundary_crossing', False)
        )
        df['contains_memory_references'] = df['lyrical_content'].apply(
            lambda x: any(l.get('contains_memory_reference', False) for l in x)
        )
        df['genre_instability_score'] = df.apply(
            lambda row: self._calculate_genre_instability_score(row), axis=1
        )

        if mm_audio_path:
            df['audio_energy'] = df['audio_features'].apply(lambda x: x.get('rms_energy', 0))
            df['spectral_complexity'] = df['audio_features'].apply(lambda x: x.get('spectral_centroid', 0))
            df['audio_rebracketing_intensity'] = df['audio_rebracketing_metrics'].apply(
                lambda x: (x.get('spectral_instability', 0) + x.get('temporal_discontinuity', 0)) / 2
            )

        if mm_midi_path:
            df['midi_density'] = df['midi_features'].apply(lambda x: x.get('note_density', 0))
            df['pitch_complexity'] = df['midi_features'].apply(lambda x: x.get('pitch_variety', 0))
            df['harmonic_rebracketing_intensity'] = df['midi_rebracketing_metrics'].apply(
                lambda x: x.get('harmonic_discontinuity', 0)
            )

        print(f"\n=== ENHANCED REBRACKETING ANALYSIS ===")
        print(f"Total segments: {len(df)}")
        print(f"Segments following Orange methodology: {df['follows_orange_methodology'].sum()}")
        print(f"Segments with memory boundary crossing: {df['memory_boundary_crossing'].sum()}")
        print(f"Segments with temporal bleeding: {df['has_temporal_bleeding'].sum()}")
        print(f"Average rebracketing score: {df['comprehensive_rebracketing_score'].mean():.4f}")
        print(f"Genre instability average: {df['genre_instability_score'].mean():.4f}")

        if mm_audio_path:
            print(f"Average audio rebracketing intensity: {df['audio_rebracketing_intensity'].mean():.4f}")
        if mm_midi_path:
            print(f"Average harmonic rebracketing intensity: {df['harmonic_rebracketing_intensity'].mean():.4f}")

        return df

    # Helper methods for rebracketing analysis

    def _check_memory_references(self, text: str) -> bool:
        """Check if lyrics contain memory-related language"""
        memory_markers = ['remember', 'recall', 'memory', 'forget', 'used to', 'back then', 'was', 'were']
        return any(marker in text.lower() for marker in memory_markers)

    def _check_temporal_markers(self, text: str) -> bool:
        """Check if lyrics contain temporal boundary language"""
        temporal_markers = ['time', 'then', 'now', 'when', 'once', 'before', 'after', 'years', 'ago']
        return any(marker in text.lower() for marker in temporal_markers)

    def _check_rebracketing_language(self, text: str) -> bool:
        """Check if lyrics contain language suggesting boundary dissolution"""
        rebracketing_markers = ['shift', 'change', 'blur', 'fade', 'cross', 'between', 'boundary', 'edge']
        return any(marker in text.lower() for marker in rebracketing_markers)

    def _calculate_lyrical_boundary_fluidity(self, temporal_rel: str, confidence: float) -> float:
        """Calculate how much lyrical content exhibits boundary fluidity"""
        base_fluidity = 0.0

        if temporal_rel in ['bleeds_in', 'bleeds_out', 'spans_across']:
            base_fluidity = 0.7
        elif temporal_rel == 'contained':
            base_fluidity = 0.3
        else:
            base_fluidity = 0.5

        # Adjust by confidence - low confidence suggests more boundary ambiguity
        confidence_adjustment = (1.0 - confidence) * 0.3

        return min(base_fluidity + confidence_adjustment, 1.0)

    def _calculate_section_rebracketing_score(self, section: dict, lyrics: list, concept_analysis) -> float:
        """Calculate rebracketing intensity for this specific section"""
        score = 0.0

        # Base score from concept analysis
        if concept_analysis:
            score += concept_analysis.memory_discrepancy_severity * 0.4

        # Score from lyrical boundary fluidity
        if lyrics:
            avg_fluidity = sum(l.get('lyrical_boundary_fluidity', 0) for l in lyrics) / len(lyrics)
            score += avg_fluidity * 0.3

        # Score from section description complexity
        description = section.get('description', '')
        complexity_indicators = ['shift', 'suddenly', 'now', 'change', 'different', 'transition']
        description_complexity = sum(0.05 for indicator in complexity_indicators if indicator in description.lower())
        score += min(description_complexity, 0.3)

        return min(score, 1.0)

    def _identify_boundary_crossing_indicators(self, section: dict, lyrics: list) -> list:
        """Identify specific indicators of temporal/modal boundary crossing in this section"""
        indicators = []

        # From section description
        description = section.get('description', '').lower()
        if 'suddenly' in description or 'shift' in description:
            indicators.append('abrupt_musical_transition')
        if 'different' in description or 'change' in description:
            indicators.append('modal_transformation')

        # From lyrical content
        if lyrics:
            memory_refs = sum(1 for l in lyrics if l.get('contains_memory_reference', False))
            if memory_refs > 0:
                indicators.append('memory_reference_clustering')

            temporal_refs = sum(1 for l in lyrics if l.get('contains_temporal_markers', False))
            if temporal_refs > 1:
                indicators.append('temporal_marker_density')

        return indicators

    def _calculate_genre_instability_score(self, row) -> float:
        """Calculate genre instability score from audio and MIDI features"""
        score = 0.0

        # Audio-based genre instability
        if 'audio_rebracketing_metrics' in row and row['audio_rebracketing_metrics']:
            audio_metrics = row['audio_rebracketing_metrics']
            score += audio_metrics.get('spectral_instability', 0) * 0.3
            score += audio_metrics.get('temporal_discontinuity', 0) * 0.3

        # MIDI-based genre instability
        if 'midi_rebracketing_metrics' in row and row['midi_rebracketing_metrics']:
            midi_metrics = row['midi_rebracketing_metrics']
            score += midi_metrics.get('harmonic_discontinuity', 0) * 0.4

        return min(score, 1.0)

    def _calculate_comprehensive_rebracketing_score(self, segment: Dict[str, Any]) -> float:
        """Calculate comprehensive rebracketing score across all modalities"""
        score = 0.0

        # Lyrical rebracketing score
        if segment.get('lyrical_content'):
            lyric_scores = [l.get('lyrical_boundary_fluidity', 0) for l in segment['lyrical_content']]
            if lyric_scores:
                score += sum(lyric_scores) / len(lyric_scores) * 0.3

        # Audio rebracketing score
        if segment.get('audio_rebracketing_metrics'):
            audio_metrics = segment['audio_rebracketing_metrics']
            audio_score = (audio_metrics.get('spectral_instability', 0) +
                          audio_metrics.get('temporal_discontinuity', 0)) / 2
            score += audio_score * 0.35

        # MIDI rebracketing score
        if segment.get('midi_rebracketing_metrics'):
            midi_metrics = segment['midi_rebracketing_metrics']
            midi_score = midi_metrics.get('harmonic_discontinuity', 0)
            score += midi_score * 0.35

        return min(score, 1.0)

    def _calculate_spectral_instability(self, audio_features: Dict[str, Any]) -> float:
        """Calculate spectral instability from audio features"""
        # Simple proxy for spectral instability
        spectral_centroid = audio_features.get('spectral_centroid', 0)
        rms_energy = audio_features.get('rms_energy', 0)

        # Higher spectral centroid + energy variation suggests instability
        instability = min((spectral_centroid / 4000.0) * (rms_energy * 10), 1.0)
        return instability

    def _calculate_temporal_discontinuity(self, audio_features: Dict[str, Any]) -> float:
        """Calculate temporal discontinuity from audio features"""
        # Use attack time as proxy for temporal discontinuity
        attack_time = audio_features.get('attack_time', 0)

        # Very fast attack suggests abrupt changes
        discontinuity = max(0, 1.0 - (attack_time * 10))
        return min(discontinuity, 1.0)

    def _detect_genre_shift_indicators(self, audio_features: Dict[str, Any], section: Dict) -> List[str]:
        """Detect indicators of genre shifts in audio"""
        indicators = []

        spectral_centroid = audio_features.get('spectral_centroid', 0)
        rms_energy = audio_features.get('rms_energy', 0)

        if spectral_centroid > 3000:
            indicators.append('high_frequency_emphasis')
        if rms_energy > 0.5:
            indicators.append('high_energy_density')
        if audio_features.get('attack_time', 0) < 0.05:
            indicators.append('percussive_elements')

        return indicators

    def _calculate_harmonic_discontinuity(self, midi_features: Dict[str, Any]) -> float:
        """Calculate harmonic discontinuity from MIDI features"""
        pitch_variety = midi_features.get('pitch_variety', 0)
        rhythmic_regularity = midi_features.get('rhythmic_regularity', 1)

        # High pitch variety + low rhythmic regularity suggests discontinuity
        discontinuity = pitch_variety * (1.0 - rhythmic_regularity)
        return min(discontinuity, 1.0)

    def _calculate_rhythmic_rebracketing(self, midi_features: Dict[str, Any]) -> float:
        """Calculate rhythmic rebracketing intensity from MIDI features"""
        rhythmic_regularity = midi_features.get('rhythmic_regularity', 1)
        note_density = midi_features.get('note_density', 0)

        # Low regularity + high density suggests rebracketing
        rebracketing = (1.0 - rhythmic_regularity) * min(note_density / 5.0, 1.0)
        return min(rebracketing, 1.0)

    def _detect_instrumental_genre_mixing(self, midi_features: Dict[str, Any]) -> List[str]:
        """Detect instrumental genre mixing patterns"""
        indicators = []

        avg_polyphony = midi_features.get('avg_polyphony', 0)
        note_density = midi_features.get('note_density', 0)
        pitch_variety = midi_features.get('pitch_variety', 0)

        if avg_polyphony > 3:
            indicators.append('complex_harmonic_texture')
        if note_density > 4:
            indicators.append('dense_rhythmic_activity')
        if pitch_variety > 0.7:
            indicators.append('wide_pitch_range')

        return indicators

    def save_training_data(self, df: pd.DataFrame, base_path: str, format: str = 'parquet') -> Dict[str, str]:
        """
        Save multimodal training data in various formats optimized for ML training

        Args:
            df: The multimodal segments DataFrame
            base_path: Base path for saving files (without extension)
            format: 'parquet', 'pickle', 'hybrid', or 'all'

        Returns:
            Dictionary with paths of saved files
        """
        saved_files = {}

        if format in ['parquet', 'hybrid', 'all']:
            # Parquet: Best for tabular data, fast loading, cross-platform
            parquet_path = f"{base_path}_training_data.parquet"

            # Flatten complex objects for parquet storage
            df_flat = self._flatten_for_parquet(df.copy())
            df_flat.to_parquet(parquet_path, compression='snappy', index=False)
            saved_files['parquet'] = parquet_path
            print(f"✅ Saved flattened training data to: {parquet_path}")

        if format in ['pickle', 'hybrid', 'all']:
            # Pickle: Preserves exact Python objects, including numpy arrays
            pickle_path = f"{base_path}_full_objects.pkl"
            df.to_pickle(pickle_path, compression='gzip')
            saved_files['pickle'] = pickle_path
            print(f"✅ Saved full objects to: {pickle_path}")

        if format in ['hybrid', 'all']:
            # JSON: Human readable metadata and structure
            def convert_numpy_types(obj):
                """Convert numpy types to native Python types for JSON serialization"""
                if hasattr(obj, 'item'):  # numpy scalars
                    return obj.item()
                elif hasattr(obj, 'tolist'):  # numpy arrays
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, (list, tuple)):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                else:
                    return obj

            metadata = {
                'manifest_id': self.manifest_id,
                'creation_time': pd.Timestamp.now().isoformat(),
                'total_segments': len(df),
                'columns': list(df.columns),
                'segment_types': df['segment_type'].unique().tolist() if 'segment_type' in df.columns else [],
                'has_audio': int(df['audio_features'].notna().sum()) if 'audio_features' in df.columns else 0,
                'has_midi': int(df['midi_features'].notna().sum()) if 'midi_features' in df.columns else 0,
                'has_lyrics': int(df['lyric_count'].sum()) if 'lyric_count' in df.columns else 0,
            }

            # Convert any numpy types to native Python types
            metadata = convert_numpy_types(metadata)

            json_path = f"{base_path}_metadata.json"
            import json
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            saved_files['metadata'] = json_path
            print(f"✅ Saved metadata to: {json_path}")

        return saved_files

    def load_training_data(self, base_path: str, format: str = 'parquet') -> pd.DataFrame:
        """
        Load multimodal training data from saved files

        Args:
            base_path: Base path used when saving
            format: 'parquet' (flattened) or 'pickle' (full objects)

        Returns:
            Loaded DataFrame
        """
        if format == 'parquet':
            parquet_path = f"{base_path}_training_data.parquet"
            df = pd.read_parquet(parquet_path)
            print(f"✅ Loaded training data from: {parquet_path}")
            return self._unflatten_from_parquet(df)

        elif format == 'pickle':
            pickle_path = f"{base_path}_full_objects.pkl"
            df = pd.read_pickle(pickle_path)
            print(f"✅ Loaded full objects from: {pickle_path}")
            return df

        else:
            raise ValueError("Format must be 'parquet' or 'pickle'")

    def _flatten_for_parquet(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flatten complex objects for parquet storage"""
        df_flat = df.copy()

        # Convert numpy arrays to JSON strings for parquet compatibility
        array_columns = ['mfcc', 'chroma', 'spectral_contrast', 'onset_frames',
                        'onset_strength', 'decay_profile']

        for col in array_columns:
            if col in df_flat.columns:
                df_flat[col] = df_flat[col].apply(
                    lambda x: x.tolist() if hasattr(x, 'tolist') else x
                )

        # Convert TimeSignature and other music21 objects to strings
        if 'time_signature' in df_flat.columns:
            df_flat['time_signature'] = df_flat['time_signature'].apply(
                lambda x: str(x) if x is not None else None
            )

        # Convert Key objects to strings if present
        if 'key' in df_flat.columns:
            df_flat['key'] = df_flat['key'].apply(
                lambda x: str(x) if x is not None else None
            )

        # Convert RainbowTableColor objects to strings
        if 'rainbow_color' in df_flat.columns:
            df_flat['rainbow_color'] = df_flat['rainbow_color'].apply(
                lambda x: str(x) if x is not None else None
            )

        # Convert any other complex objects that might cause issues
        for col in df_flat.columns:
            if df_flat[col].dtype == 'object':
                # Check if any values in this column are complex objects
                sample_val = df_flat[col].dropna().iloc[0] if len(df_flat[col].dropna()) > 0 else None
                if sample_val is not None and hasattr(sample_val, '__class__') and hasattr(sample_val.__class__, '__module__'):
                    module_name = sample_val.__class__.__module__
                    if module_name and ('music21' in module_name or 'pretty_midi' in module_name or 'app.structures' in module_name):
                        df_flat[col] = df_flat[col].apply(
                            lambda x: str(x) if x is not None else None
                        )

        # Flatten nested dictionaries
        if 'audio_features' in df_flat.columns:
            df_flat['audio_rms_energy'] = df_flat['audio_features'].apply(
                lambda x: x.get('rms_energy', 0) if isinstance(x, dict) else 0
            )
            df_flat['audio_spectral_centroid'] = df_flat['audio_features'].apply(
                lambda x: x.get('spectral_centroid', 0) if isinstance(x, dict) else 0
            )
            df_flat['audio_attack_time'] = df_flat['audio_features'].apply(
                lambda x: x.get('attack_time', 0) if isinstance(x, dict) else 0
            )
            # Remove original complex column
            df_flat = df_flat.drop('audio_features', axis=1)

        if 'midi_features' in df_flat.columns:
            df_flat['midi_note_density'] = df_flat['midi_features'].apply(
                lambda x: x.get('note_density', 0) if isinstance(x, dict) else 0
            )
            df_flat['midi_pitch_variety'] = df_flat['midi_features'].apply(
                lambda x: x.get('pitch_variety', 0) if isinstance(x, dict) else 0
            )
            df_flat['midi_avg_polyphony'] = df_flat['midi_features'].apply(
                lambda x: x.get('avg_polyphony', 0) if isinstance(x, dict) else 0
            )
            df_flat = df_flat.drop('midi_features', axis=1)

        # Flatten complex nested dictionaries and lists that might contain objects
        complex_columns = ['rebracketing_features', 'concept_analysis', 'audio_rebracketing_metrics',
                          'midi_rebracketing_metrics', 'boundary_crossing_indicators']

        for col in complex_columns:
            if col in df_flat.columns:
                df_flat[f'{col}_json'] = df_flat[col].apply(
                    lambda x: str(x) if x is not None else ""
                )
                df_flat = df_flat.drop(col, axis=1)

        # Convert lyrical_content to string representation for parquet
        if 'lyrical_content' in df_flat.columns:
            import json
            df_flat['lyrical_content_json'] = df_flat['lyrical_content'].apply(
                lambda x: json.dumps(x) if x else ""
            )
            df_flat = df_flat.drop('lyrical_content', axis=1)

        return df_flat

    def _unflatten_from_parquet(self, df_flat: pd.DataFrame) -> pd.DataFrame:
        """Reconstruct complex objects from flattened parquet data"""
        df = df_flat.copy()

        # Reconstruct audio_features dict
        audio_cols = [col for col in df.columns if col.startswith('audio_')]
        if audio_cols:
            df['audio_features'] = df.apply(lambda row: {
                col.replace('audio_', ''): row[col] for col in audio_cols
            }, axis=1)
            df = df.drop(audio_cols, axis=1)

        # Reconstruct midi_features dict
        midi_cols = [col for col in df.columns if col.startswith('midi_')]
        if midi_cols:
            df['midi_features'] = df.apply(lambda row: {
                col.replace('midi_', ''): row[col] for col in midi_cols
            }, axis=1)
            df = df.drop(midi_cols, axis=1)

        # Reconstruct lyrical_content from JSON string
        if 'lyrical_content_json' in df.columns:
            import json
            def safe_json_loads(x):
                if not x or x == "":
                    return []
                try:
                    return json.loads(x)
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    print(f"Warning: Could not parse JSON for lyrical_content: {e}")
                    # Try to handle it as a string representation
                    try:
                        import ast
                        return ast.literal_eval(x)
                    except (ValueError, SyntaxError):
                        print(f"Warning: Could not parse as literal either, returning empty list")
                        return []  # Add missing return statement

            df['lyrical_content'] = df['lyrical_content_json'].apply(safe_json_loads)
            df = df.drop('lyrical_content_json', axis=1)

        return df

    # ...existing methods...

if __name__ == "__main__":
    import os
    import pandas as pd
    print("=== ManifestExtractor Example Usage ===")
    # Set up environment for manifest loading
    os.environ['MANIFEST_PATH'] = '/Volumes/LucidNonsense/White/staged_raw_material'
    manifest_id = "01_01"
    yaml_path = f"{os.getenv('MANIFEST_PATH')}/{manifest_id}/{manifest_id}.yml"
    lrc_path = f"{os.getenv('MANIFEST_PATH')}/{manifest_id}/{manifest_id}.lrc"
    audio_path = f"{os.getenv('MANIFEST_PATH')}/{manifest_id}/01_01_02.wav"
    midi_path = f"{os.getenv('MANIFEST_PATH')}/{manifest_id}/01_01_biotron.mid"

    # Initialize extractor
    extractor = ManifestExtractor(mani_id=manifest_id)
    print("Loaded manifest:", extractor.manifest.title if extractor.manifest else "None")

    # Example: generate_multimodal_segments (if files exist)
    if os.path.exists(yaml_path):
        try:
            df = extractor.generate_multimodal_segments(yaml_path, lrc_path, audio_path, midi_path)
            print("\nGenerated multimodal segments DataFrame:")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")

            # Save training data in different formats
            base_save_path = f"/Volumes/LucidNonsense/White/rainbow-pipeline/data/{manifest_id}_multimodal"

            # Save in all formats for comparison
            saved_files = extractor.save_training_data(df, base_save_path, format='all')

            print(f"\n=== SAVED TRAINING DATA ===")
            for format_type, file_path in saved_files.items():
                file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
                print(f"{format_type:>10}: {file_path} ({file_size:.2f} MB)")

            # Test loading data back
            print(f"\n=== LOADING TEST ===")
            df_loaded_parquet = extractor.load_training_data(base_save_path, format='parquet')
            df_loaded_pickle = extractor.load_training_data(base_save_path, format='pickle')

            print(f"Original shape: {df.shape}")
            print(f"Parquet loaded shape: {df_loaded_parquet.shape}")
            print(f"Pickle loaded shape: {df_loaded_pickle.shape}")

        except Exception as e:
            print(f"generate_multimodal_segments error: {e}")
    else:
        print(f"Manifest YAML not found: {yaml_path}")
