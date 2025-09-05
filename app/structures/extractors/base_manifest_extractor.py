import numpy as np
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Dict
from pydantic import BaseModel

from app.structures.enums.temporal_relatioship import TemporalRelationship
from app.structures.manifests.manifest import Manifest
from app.util.manifest_loader import load_manifest
from app.util.manifest_validator import validate_yaml_file

class BaseManifestExtractor(BaseModel):

    manifest: Manifest | None = None
    manifest_path: str | None = None
    manifest_id: str | None = None

    def __init__(self, **data):
        load_dotenv()
        super().__init__(**data)
        self.manifest_path = (
            os.path.join(os.environ['MANIFEST_PATH'],
            self.manifest_id,
            f"{self.manifest_id}.yml"
            ) if self.manifest_id else None
        )
        if not self.manifest_path:
            raise ValueError("manifest_id must be provided.")
        all_valid, errors = validate_yaml_file(self.manifest_path)
        if all_valid:
            print("YAML file valid.")
        else:
            print("YAML file has errors:")
            for error in errors:
                print(f" - {error}")
        self.manifest = load_manifest(self.manifest_path) if self.manifest_path else None
        if self.manifest  is None:
            raise ValueError("Manifest could not be loaded. Please provide a valid manifest_path.")


    def parse_yaml_time(self, time_str: str) -> float:
        """Parse YAML timestamp like '[00:28.086]' to seconds"""
        return self.parse_lrc_time(time_str)


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

    def generate_multimodal_segments(self, yaml_path: str, lrc_path: str = None,
                                 audio_path: str = None, midi_path: str = None) -> pd.DataFrame:
        """Generate complete multimodal training dataset"""
        print(f"=== MULTIMODAL REBRACKETING TRAINING DATA GENERATION ===")
        print(f"YAML: {yaml_path}")
        print(f"LRC:  {lrc_path}")
        print(f"Audio: {audio_path}")
        print(f"MIDI: {midi_path}")

        manifest = self.load_manifest(yaml_path)
        print(f"✅ Loaded manifest: {manifest.get('title', 'Unknown')}")

        # Load lyrics if provided
        lyrics = []
        if lrc_path and Path(lrc_path).exists():
            lyrics = self.load_lrc(lrc_path)
            print(f"✅ Loaded {len(lyrics)} lyrics")

        all_segments = []

        # Generate section-based segments with multimodal data
        for section in manifest['structure']:
            section_start = self.parse_yaml_time(section['start_time'])
            section_end = self.parse_yaml_time(section['end_time'])

            # Find intersecting lyrics
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

                        intersecting_lyrics.append({
                            'text': lyric['text'],
                            'start_time': lyric_start,
                            'end_time': lyric_end,
                            'temporal_relationship': temporal_rel,
                            'confidence': confidence,
                            'overlap_seconds': overlap_duration
                        })
            # ToDo: new class for Segment
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
                'bpm': manifest.get('bpm'),
                'time_signature': manifest.get('tempo'),
                'key': manifest.get('key'),
                'rainbow_color': manifest.get('rainbow_color'),

                # Metadata
                'title': manifest.get('title'),
                'mood_tags': manifest.get('mood', []),
                'concept': manifest.get('concept', ''),
            }

            # Add audio features if audio file provided
            if audio_path and Path(audio_path).exists():
                segment['audio_features'] = self.load_audio_segment(audio_path, section_start, section_end)
                print(f"✅ Added audio features for {section['section_name']}")

            # Add MIDI features if MIDI file provided
            if midi_path and Path(midi_path).exists():
                segment['midi_features'] = self.load_midi_segment(midi_path, section_start, section_end)
                print(f"✅ Added MIDI features for {section['section_name']}")

            # Calculate boundary fluidity metrics across all modalities
            segment['rebracketing_score'] = self._calculate_boundary_fluidity_score(segment)

            all_segments.append(segment)

        df = pd.DataFrame(all_segments)

        # Add computed multimodal metrics
        df['lyric_count'] = df['lyrical_content'].apply(len)
        df['has_temporal_bleeding'] = df['lyrical_content'].apply(
            lambda x: any(l['temporal_relationship'] not in ['contained', 'exact_match'] for l in x)
        )

        if audio_path:
            df['audio_energy'] = df['audio_features'].apply(lambda x: x.get('rms_energy', 0))
            df['spectral_complexity'] = df['audio_features'].apply(lambda x: x.get('spectral_centroid', 0))

        if midi_path:
            df['midi_density'] = df['midi_features'].apply(lambda x: x.get('note_density', 0))
            df['pitch_complexity'] = df['midi_features'].apply(lambda x: x.get('pitch_variety', 0))

        print(f"\n=== MULTIMODAL ANALYSIS ===")
        print(f"Total segments: {len(df)}")
        print(f"Segments with lyrics: {(df['lyric_count'] > 0).sum()}")
        print(f"Segments with temporal bleeding: {df['has_temporal_bleeding'].sum()}")

        if audio_path:
            print(f"Average audio energy: {df['audio_energy'].mean():.4f}")
        if midi_path:
            print(f"Average MIDI note density: {df['midi_density'].mean():.2f} notes/second")

        return df

if __name__ == "__main__":
    extractor = BaseManifestExtractor(manifest_id="01_01")
    print(extractor.manifest)