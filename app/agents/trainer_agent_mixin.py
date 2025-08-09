import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
import json

class TrainingDataDrivenMixin:
    training_data: Optional[pd.DataFrame] = None


    def find_similar_segments(self,
                              target_mood: List[str],
                              target_key: str = None,
                              target_bpm: int = None,
                              rainbow_color: str = None,
                              section_type: str = None,
                              limit: int = 10) -> pd.DataFrame:
        """Find segments from your training data that match criteria"""

        if self.training_data is None or len(self.training_data) == 0:
            print("⚠️ No training data loaded")
            return pd.DataFrame()

        # Start with all data
        filtered_data = self.training_data.copy()

        # Filter by rainbow color if specified
        if rainbow_color:
            filtered_data = filtered_data[
                filtered_data['album_rainbow_color'].str.contains(rainbow_color, case=False, na=False)
            ]

        # Filter by key if specified
        if target_key:
            filtered_data = filtered_data[
                filtered_data['song_key'].str.contains(target_key.split()[0], case=False, na=False)
            ]

        # Filter by BPM range if specified
        if target_bpm:
            bpm_tolerance = 10
            filtered_data = filtered_data[
                (pd.to_numeric(filtered_data['song_bpm'], errors='coerce') >= target_bpm - bpm_tolerance) &
                (pd.to_numeric(filtered_data['song_bpm'], errors='coerce') <= target_bpm + bpm_tolerance)
                ]

        # Filter by section type if specified
        if section_type:
            filtered_data = filtered_data[
                filtered_data['song_segment_name'].str.contains(section_type, case=False, na=False)
            ]

        # Score by mood similarity
        if target_mood and 'song_moods' in filtered_data.columns:
            filtered_data['mood_score'] = filtered_data['song_moods'].apply(
                lambda x: self._calculate_mood_similarity(target_mood, x) if pd.notna(x) else 0
            )
            # Sort by mood similarity
            filtered_data = filtered_data.sort_values('mood_score', ascending=False)

        return filtered_data.head(limit)

    @staticmethod
    def _calculate_mood_similarity(target_moods: List[str], segment_moods: str) -> float:
        """Calculate similarity between target moods and segment moods"""
        if not segment_moods or pd.isna(segment_moods):
            return 0.0

        # Parse segment moods (assuming comma-separated)
        try:
            segment_mood_list = [m.strip().lower() for m in str(segment_moods).split(',')]
            target_mood_list = [m.strip().lower() for m in target_moods]

            # Simple overlap score
            overlap = len(set(target_mood_list) & set(segment_mood_list))
            total = len(set(target_mood_list) | set(segment_mood_list))

            return overlap / max(total, 1)
        except:
            return 0.0

    def extract_musical_patterns(self, segments: pd.DataFrame) -> Dict[str, Any]:
        """Extract musical patterns from similar segments"""
        if segments.empty:
            return {}

        patterns = {
            'common_keys': segments['song_key'].value_counts().head(5).to_dict(),
            'bpm_range': {
                'min': pd.to_numeric(segments['song_bpm'], errors='coerce').min(),
                'max': pd.to_numeric(segments['song_bpm'], errors='coerce').max(),
                'avg': pd.to_numeric(segments['song_bpm'], errors='coerce').mean()
            },
            'common_structures': [],
            'common_moods': segments['song_moods'].value_counts().head(10).to_dict(),
            'genre_patterns': segments['song_genres'].value_counts().head(10).to_dict()
        }

        # Extract structure patterns
        for _, segment in segments.iterrows():
            if pd.notna(segment.get('song_structure')):
                try:
                    structure = json.loads(segment['song_structure'])
                    patterns['common_structures'].append(structure)
                except:
                    pass

        return patterns




# Test script for training data integration
if __name__ == "__main__":
    # from app.objects.rainbow_color import RainbowColor
    #
    # print("🎵 Testing Training Data-Driven Generation")
    # print("=" * 60)
    #
    # # Initialize enhanced Andy
    # andy = TrainingDataDrivenAndy()
    # andy.initialize()
    #
    # # Check training data status
    # print(f"📊 Training data loaded:")
    # print(
    #     f"  Dorthy: {len(andy.lyrics_agent.training_data) if andy.lyrics_agent.training_data is not None else 0} samples")
    # print(
    #     f"  Nancarrow: {len(andy.midi_agent.training_data) if andy.midi_agent.training_data is not None else 0} samples")
    #
    # if andy.lyrics_agent.training_data is not None and len(andy.lyrics_agent.training_data) > 0:
    #     print("\n✅ Training data found - generating from your catalog...")
    #
    #     # Generate using your actual musical catalog
    #     result = andy.generate_from_your_catalog(
    #         rainbow_color=RainbowColor.Z,
    #         target_moods=['dark', 'mysterious', 'haunting']
    #     )
    #
    #     print(f"\n🎯 Generated {len(result['sections'])} sections")
    #     for section in result['sections']:
    #         print(f"  📝 {section['section_name']}")
    #         if 'lyrics' in section:
    #             print(f"    🎤 Lyrics confidence: {section['lyrics']['confidence']:.2f}")
    #         print(f"    🎹 MIDI confidence: {section['midi']['confidence']:.2f}")
    #
    # else:
    #     print("\n⚠️ No training data found")
    #     print("💡 Run main.py first to generate training samples from your staged materials")