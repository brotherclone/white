import pandas as pd
import yaml
import re
from typing import List, Dict, Any, Tuple
from datetime import timedelta
import numpy as np


class RebracketerTrainingDataGenerator:
    """
    Generates multimodal training data that embraces temporal boundary fluidity
    Perfect for Orange album rebracketing concepts!
    """

    def __init__(self):
        self.segment_types = ['section', 'bar', 'phrase', 'sliding_window']

# Move to UT
    def parse_lrc_time(self, time_str: str) -> float:
        """Parse LRC timestamp like [00:28.085] to seconds"""
        match = re.match(r'\[(\d{2}):(\d{2})\.(\d{3})\]', time_str)
        if match:
            minutes, seconds, milliseconds = map(int, match.groups())
            return minutes * 60 + seconds + milliseconds / 1000
        return 0.0

    # Move to UT
    def parse_yaml_time(self, time_str: str) -> float:
        """Parse YAML timestamp like '[00:28.086]' to seconds"""
        return self.parse_lrc_time(time_str)

    def load_manifest(self, yaml_path: str) -> Dict[str, Any]:
        """Load the YAML manifest file"""
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)

    # Move to UT
    def load_lrc(self, lrc_path: str) -> List[Dict[str, Any]]:
        """Parse LRC file into structured lyrical content"""
        lyrics = []
        with open(lrc_path, 'r') as f:
            lines = f.readlines()

        current_time = None
        for line in lines:
            line = line.strip()
            if line.startswith('[') and ']' in line:
                # Extract timestamp
                bracket_end = line.find(']')
                timestamp = line[:bracket_end + 1]
                text = line[bracket_end + 1:].strip()

                # Skip metadata lines
                if timestamp.startswith('[ti:') or timestamp.startswith('[ar:') or timestamp.startswith('[al:'):
                    continue

                if text:  # Only process lines with actual lyrics
                    lyrics.append({
                        'text': text,
                        'start_time': self.parse_lrc_time(timestamp),
                        'timestamp_raw': timestamp
                    })

        # Calculate end times (start of next lyric or section end)
        for i in range(len(lyrics)):
            if i < len(lyrics) - 1:
                lyrics[i]['end_time'] = lyrics[i + 1]['start_time']
            else:
                lyrics[i]['end_time'] = lyrics[i]['start_time'] + 3.0  # Default 3s for last line

        return lyrics

    def determine_temporal_relationship(self, content_start: float, content_end: float,
                                        segment_start: float, segment_end: float) -> str:
        """Determine how content relates to segment boundaries"""
        if content_start < segment_start and content_end > segment_end:
            # Type or Enum
            return 'spans_across'
        elif content_start < segment_start:
            return 'bleeds_in'
        elif content_end > segment_end:
            return 'bleeds_out'
        else:
            return 'contained'

    def generate_section_segments(self, manifest: Dict[str, Any], lyrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate section-based segments with flexible lyrical content"""
        segments = []

        for section in manifest['structure']:
            section_start = self.parse_yaml_time(section['start_time'])
            section_end = self.parse_yaml_time(section['end_time'])

            # Find all lyrics that intersect with this section
            intersecting_lyrics = []
            for lyric in lyrics:
                lyric_start = lyric['start_time']
                lyric_end = lyric['end_time']

                # Check if lyric intersects with section (any overlap)
                if not (lyric_end <= section_start or lyric_start >= section_end):
                    temporal_rel = self.determine_temporal_relationship(
                        lyric_start, lyric_end, section_start, section_end
                    )

                    # Calculate confidence based on how much the lyric overlaps
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
                'rebracketing_score': len([l for l in intersecting_lyrics if l['temporal_relationship'] != 'contained'])
            }

            segments.append(segment)

        return segments

    def generate_bar_segments(self, manifest: Dict[str, Any], lyrics: List[Dict[str, Any]],
                              bar_duration: float = None) -> List[Dict[str, Any]]:
        """Generate bar-based segments for strict musical timing"""
        if not bar_duration:
            # Calculate bar duration from BPM and time signature
            bpm = manifest.get('bpm', 120)
            time_sig = manifest.get('tempo', '4/4')

            if '/' in time_sig:
                beats_per_bar = int(time_sig.split('/')[0])
                bar_duration = (beats_per_bar * 60) / bpm
            else:
                bar_duration = 240 / bpm  # Default 4/4

        # Get total track time
        trt = manifest.get('TRT', '[02:47.060]')
        total_duration = self.parse_lrc_time(trt) if trt.startswith('[') else 167.0

        segments = []
        current_time = 0.0
        bar_number = 1

        while current_time < total_duration:
            bar_end = min(current_time + bar_duration, total_duration)

            # Find intersecting lyrics
            intersecting_lyrics = []
            for lyric in lyrics:
                if not (lyric['end_time'] <= current_time or lyric['start_time'] >= bar_end):
                    temporal_rel = self.determine_temporal_relationship(
                        lyric['start_time'], lyric['end_time'], current_time, bar_end
                    )

                    overlap_start = max(lyric['start_time'], current_time)
                    overlap_end = min(lyric['end_time'], bar_end)
                    overlap_duration = max(0, overlap_end - overlap_start)
                    confidence = overlap_duration / bar_duration

                    intersecting_lyrics.append({
                        'text': lyric['text'],
                        'start_time': lyric['start_time'],
                        'end_time': lyric['end_time'],
                        'temporal_relationship': temporal_rel,
                        'confidence': confidence
                    })

            segment = {
                'manifest_id': manifest['manifest_id'],
                'segment_type': 'bar',
                'segment_id': f"{manifest['manifest_id']}_bar_{bar_number:03d}",
                'canonical_start': current_time,
                'canonical_end': bar_end,
                'duration': bar_end - current_time,
                'bar_number': bar_number,
                'lyrical_content': intersecting_lyrics,

                # Musical context
                'bpm': manifest.get('bpm'),
                'time_signature': manifest.get('tempo'),
                'key': manifest.get('key'),
                'rainbow_color': manifest.get('rainbow_color'),

                # Rebracketing metrics
                'boundary_fluidity_score': len(
                    [l for l in intersecting_lyrics if l['temporal_relationship'] != 'contained'])
            }

            segments.append(segment)
            current_time = bar_end
            bar_number += 1

        return segments

    def generate_phrase_segments(self, manifest: Dict[str, Any], lyrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate phrase-based segments that respect natural lyrical boundaries"""
        segments = []

        for i, lyric in enumerate(lyrics):
            segment = {
                'manifest_id': manifest['manifest_id'],
                'segment_type': 'phrase',
                'segment_id': f"{manifest['manifest_id']}_phrase_{i:03d}",
                'canonical_start': lyric['start_time'],
                'canonical_end': lyric['end_time'],
                'duration': lyric['end_time'] - lyric['start_time'],
                'phrase_number': i + 1,
                'lyrical_content': [{
                    'text': lyric['text'],
                    'start_time': lyric['start_time'],
                    'end_time': lyric['end_time'],
                    'temporal_relationship': 'exact_match',
                    'confidence': 1.0
                }],

                # Find which section this phrase belongs to
                'parent_section': self._find_parent_section(lyric, manifest),

                # Musical context
                'bpm': manifest.get('bpm'),
                'time_signature': manifest.get('tempo'),
                'key': manifest.get('key'),
                'rainbow_color': manifest.get('rainbow_color')
            }

            segments.append(segment)

        return segments

    def _find_parent_section(self, lyric: Dict[str, Any], manifest: Dict[str, Any]) -> str:
        """Find which section a lyric belongs to"""
        lyric_time = lyric['start_time']

        for section in manifest['structure']:
            section_start = self.parse_yaml_time(section['start_time'])
            section_end = self.parse_yaml_time(section['end_time'])

            if section_start <= lyric_time < section_end:
                return section['section_name']

        return 'unknown'

    def generate_all_segments(self, yaml_path: str, lrc_path: str) -> pd.DataFrame:
        """Generate complete training dataset with all segment types"""
        manifest = self.load_manifest(yaml_path)
        lyrics = self.load_lrc(lrc_path)

        all_segments = []

        # Generate different segment types
        all_segments.extend(self.generate_section_segments(manifest, lyrics))
        all_segments.extend(self.generate_bar_segments(manifest, lyrics))
        all_segments.extend(self.generate_phrase_segments(manifest, lyrics))

        df = pd.DataFrame(all_segments)

        # Add some computed rebracketing metrics
        df['lyric_count'] = df['lyrical_content'].apply(len)
        df['has_temporal_bleeding'] = df['lyrical_content'].apply(
            lambda x: any(
                l['temporal_relationship'] != 'contained' and l['temporal_relationship'] != 'exact_match' for l in x)
        )
        df['avg_lyric_confidence'] = df['lyrical_content'].apply(
            lambda x: np.mean([l['confidence'] for l in x]) if x else 0
        )

        return df


# Example usage:
if __name__ == "__main__":
    generator = RebracketerTrainingDataGenerator()

    # Generate training data
    df = generator.generate_all_segments('staged_raw_material/03_01/03_01.yml',
                                         'staged_raw_material/03_01/03_01.lrc')

    print(f"Generated {len(df)} training segments")
    print(f"Segment types: {df['segment_type'].value_counts()}")
    print(f"Segments with temporal bleeding: {df['has_temporal_bleeding'].sum()}")

    # Save to parquet for efficient training data access
    df.to_parquet('moonrakers_training_data.parquet', index=False)

    # Show a sample of rebracketed segments
    print("\n=== REBRACKETING IN ACTION ===")
    bleeding_segments = df[df['has_temporal_bleeding'] == True].head(3)

    for _, segment in bleeding_segments.iterrows():
        print(f"\nSegment: {segment['segment_id']}")
        print(f"Official boundaries: {segment['canonical_start']:.3f}s - {segment['canonical_end']:.3f}s")
        print("Lyrical content:")
        for lyric in segment['lyrical_content']:
            print(f"  '{lyric['text']}' ({lyric['start_time']:.3f}s-{lyric['end_time']:.3f}s)")
            print(f"    Relationship: {lyric['temporal_relationship']}, Confidence: {lyric['confidence']:.2f}")