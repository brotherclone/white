# import yaml
# import librosa
# import numpy as np
# from pydub import AudioSegment
# import pandas as pd
# import json
# from pathlib import Path
#
#
# class MusicDataExtractor:
#     def __init__(self, yaml_file_path):
#         """Initialize with your track metadata YAML file"""
#         with open(yaml_file_path, 'r') as f:
#             self.metadata = yaml.safe_load(f)
#
#     def extract_section_data(self, section_name="Verse 1"):
#         """Extract all data for a specific section"""
#
#         # Find the section timing
#         section_info = None
#         for section in self.metadata['structure']:
#             if section['section'] == section_name:
#                 section_info = section
#                 break
#
#         if not section_info:
#             raise ValueError(f"Section '{section_name}' not found")
#
#         # Convert timestamps to seconds
#         start_time = self._timestamp_to_seconds(section_info['start_time'])
#         end_time = self._timestamp_to_seconds(section_info['end_time'])
#         duration = end_time - start_time
#
#         print(f"Extracting {section_name}: {start_time:.3f}s to {end_time:.3f}s ({duration:.3f}s)")
#
#         extracted_data = {
#             'section_info': section_info,
#             'timing': {
#                 'start_seconds': start_time,
#                 'end_seconds': end_time,
#                 'duration_seconds': duration
#             },
#             'audio_chunks': {},
#             'lyrics': None,
#             'midi_data': {},
#             'active_instruments': []
#         }
#
#         # Extract audio chunks for each track
#         self._extract_audio_chunks(extracted_data, start_time, end_time)
#
#         # Extract lyrics for this timeframe
#         self._extract_lyrics(extracted_data, start_time, end_time)
#
#         # Extract MIDI data
#         self._extract_midi_data(extracted_data, start_time, end_time)
#
#         return extracted_data
#
#     def _timestamp_to_seconds(self, timestamp_str):
#         """Convert MM:SS.sss to seconds"""
#         if isinstance(timestamp_str, (int, float)):
#             return float(timestamp_str)
#
#         parts = timestamp_str.split(':')
#         minutes = int(parts[0])
#         seconds = float(parts[1])
#         return minutes * 60 + seconds
#
#     def _extract_audio_chunks(self, extracted_data, start_time, end_time):
#         """Extract audio chunks from all tracks"""
#         start_ms = int(start_time * 1000)
#         end_ms = int(end_time * 1000)
#
#         for track_name, track_info in self.metadata['audio_tracks'].items():
#             audio_file = track_info['audio_file']
#
#             try:
#                 # Load audio file
#                 audio = AudioSegment.from_wav(audio_file)
#
#                 # Extract the section
#                 chunk = audio[start_ms:end_ms]
#
#                 # Save extracted chunk
#                 output_file = f"{track_name}_verse1_chunk.wav"
#                 chunk.export(output_file, format="wav")
#
#                 extracted_data['audio_chunks'][track_name] = {
#                     'file': output_file,
#                     'description': track_info['description'],
#                     'original_file': audio_file,
#                     'duration_ms': len(chunk),
#                     'has_audio': len(chunk) > 0
#                 }
#
#                 # Check if this track is active (has significant audio)
#                 if self._has_significant_audio(chunk):
#                     extracted_data['active_instruments'].append({
#                         'track': track_name,
#                         'description': track_info['description'],
#                         'id': track_info['id']
#                     })
#
#                 print(f"✓ Extracted {track_name}: {len(chunk) / 1000:.2f}s")
#
#             except Exception as e:
#                 print(f"✗ Failed to extract {track_name}: {e}")
#                 extracted_data['audio_chunks'][track_name] = {
#                     'error': str(e),
#                     'original_file': audio_file
#                 }
#
#     def _has_significant_audio(self, audio_chunk, threshold_db=-40):
#         """Check if audio chunk has significant content"""
#         if len(audio_chunk) == 0:
#             return False
#
#         # Convert to numpy array and check RMS level
#         samples = np.array(audio_chunk.get_array_of_samples())
#         if len(samples) == 0:
#             return False
#
#         rms = np.sqrt(np.mean(samples ** 2))
#         db_level = 20 * np.log10(rms + 1e-10)  # Avoid log(0)
#
#         return db_level > threshold_db
#
#     def _extract_lyrics(self, extracted_data, start_time, end_time):
#         """Extract lyrics for this time range from LRC file"""
#         lrc_file = self.metadata.get('lrc_file')
#         if not lrc_file:
#             print("No LRC file specified")
#             return
#
#         try:
#             lyrics_in_section = []
#
#             with open(lrc_file, 'r', encoding='utf-8') as f:
#                 for line in f:
#                     line = line.strip()
#                     if line and line.startswith('[') and ']' in line:
#                         # Parse LRC timestamp [mm:ss.xx]
#                         timestamp_end = line.find(']')
#                         timestamp_str = line[1:timestamp_end]
#                         lyric_text = line[timestamp_end + 1:].strip()
#
#                         if lyric_text:  # Skip empty lines
#                             lyric_time = self._lrc_timestamp_to_seconds(timestamp_str)
#
#                             if start_time <= lyric_time <= end_time:
#                                 lyrics_in_section.append({
#                                     'time': lyric_time,
#                                     'relative_time': lyric_time - start_time,
#                                     'text': lyric_text
#                                 })
#
#             extracted_data['lyrics'] = sorted(lyrics_in_section, key=lambda x: x['time'])
#             print(f"✓ Extracted {len(lyrics_in_section)} lyric lines")
#
#         except Exception as e:
#             print(f"✗ Failed to extract lyrics: {e}")
#             extracted_data['lyrics'] = []
#
#     def _lrc_timestamp_to_seconds(self, timestamp_str):
#         """Convert LRC timestamp mm:ss.xx to seconds"""
#         try:
#             parts = timestamp_str.split(':')
#             minutes = int(parts[0])
#             seconds = float(parts[1])
#             return minutes * 60 + seconds
#         except:
#             return 0
#
#     def _extract_midi_data(self, extracted_data, start_time, end_time):
#         """Extract MIDI data for this time range"""
#         # This would need a MIDI library like pretty_midi or mido
#         # Placeholder for now
#         extracted_data['midi_data'] = {
#             'note': 'MIDI extraction would go here with pretty_midi library'
#         }
#
#     def create_training_sample(self, section_data):
#         """Create a structured training sample from extracted data"""
#
#         training_sample = {
#             'metadata': {
#                 'title': self.metadata['title'],
#                 'artist': self.metadata['artist'],
#                 'bpm': self.metadata['bpm'],
#                 'key': self.metadata['key'],
#                 'rainbow_color': self.metadata['rainbow_color'],
#                 'mood': self.metadata['mood'],
#                 'genres': self.metadata['genres']
#             },
#             'section': {
#                 'name': section_data['section_info']['section'],
#                 'description': section_data['section_info']['description'],
#                 'duration': section_data['timing']['duration_seconds'],
#                 'start_time': section_data['timing']['start_seconds']
#             },
#             'instrumentation': section_data['active_instruments'],
#             'lyrics': section_data['lyrics'],
#             'audio_files': {k: v['file'] for k, v in section_data['audio_chunks'].items() if 'file' in v},
#             'temporal_structure': self._create_temporal_structure(section_data)
#         }
#
#         return training_sample
#
#     def save_training_data(self, training_sample, output_dir="training_data"):
#         """Save training data in multiple optimized formats"""
#         import os
#         os.makedirs(output_dir, exist_ok=True)
#
#         # 1. JSON for rich metadata and structure
#         with open(f"{output_dir}/verse1_metadata.json", 'w') as f:
#             json.dump(training_sample, f, indent=2)
#
#         # 2. Parquet for temporal/tabular data
#         temporal_df = self._create_temporal_dataframe(training_sample)
#         temporal_df.to_parquet(f"{output_dir}/verse1_temporal.parquet")
#
#         # 3. Text embeddings file for semantic search
#         text_data = self._extract_text_for_embeddings(training_sample)
#         with open(f"{output_dir}/verse1_text.txt", 'w') as f:
#             f.write(text_data)
#
#         # 4. Audio file manifest
#         audio_manifest = pd.DataFrame([
#             {
#                 'track_name': track,
#                 'file_path': path,
#                 'description': training_sample['instrumentation'][i].get('description', '') if i < len(
#                     training_sample['instrumentation']) else '',
#                 'duration': training_sample['section']['duration']
#             }
#             for i, (track, path) in enumerate(training_sample['audio_files'].items())
#         ])
#         audio_manifest.to_parquet(f"{output_dir}/verse1_audio_manifest.parquet")
#
#         print(f"✓ Saved training data to {output_dir}/")
#         return output_dir
#
#     def _create_temporal_dataframe(self, training_sample):
#         """Create a time-series DataFrame for analysis"""
#         events = []
#
#         # Add lyrical events
#         if training_sample['lyrics']:
#             for lyric in training_sample['lyrics']:
#                 events.append({
#                     'timestamp': lyric['relative_time'],
#                     'event_type': 'lyric',
#                     'content': lyric['text'],
#                     'section': training_sample['section']['name'],
#                     'bpm': training_sample['metadata']['bpm'],
#                     'key': training_sample['metadata']['key'],
#                     'rainbow_color': training_sample['metadata']['rainbow_color']
#                 })
#
#         # Add instrument events
#         for instrument in training_sample['instrumentation']:
#             events.append({
#                 'timestamp': 0.0,  # Start of section
#                 'event_type': 'instrument_active',
#                 'content': instrument['description'],
#                 'section': training_sample['section']['name'],
#                 'bpm': training_sample['metadata']['bpm'],
#                 'key': training_sample['metadata']['key'],
#                 'rainbow_color': training_sample['metadata']['rainbow_color']
#             })
#
#         # Add mood tags as events
#         for mood in training_sample['metadata']['mood']:
#             events.append({
#                 'timestamp': 0.0,
#                 'event_type': 'mood',
#                 'content': mood,
#                 'section': training_sample['section']['name'],
#                 'bpm': training_sample['metadata']['bpm'],
#                 'key': training_sample['metadata']['key'],
#                 'rainbow_color': training_sample['metadata']['rainbow_color']
#             })
#
#         df = pd.DataFrame(events)
#         df = df.sort_values('timestamp').reset_index(drop=True)
#         return df
#
#     def _extract_text_for_embeddings(self, training_sample):
#         """Extract all text content for semantic embeddings"""
#         text_parts = []
#
#         # Section description
#         text_parts.append(f"Section: {training_sample['section']['description']}")
#
#         # Lyrics
#         if training_sample['lyrics']:
#             lyrics_text = " ".join([lyric['text'] for lyric in training_sample['lyrics']])
#             text_parts.append(f"Lyrics: {lyrics_text}")
#
#         # Instrument descriptions
#         instruments = " ".join([inst['description'] for inst in training_sample['instrumentation']])
#         text_parts.append(f"Instruments: {instruments}")
#
#         # Mood and genre
#         moods = " ".join(training_sample['metadata']['mood'])
#         genres = " ".join(training_sample['metadata']['genres'])
#         text_parts.append(f"Mood: {moods}")
#         text_parts.append(f"Genres: {genres}")
#
#         # Color association
#         text_parts.append(f"Color: {training_sample['metadata']['rainbow_color']}")
#
#         return "\n".join(text_parts)
#
#     def _create_temporal_structure(self, section_data):
#         """Create a timeline of all events in this section"""
#         events = []
#
#         # Add lyric events
#         if section_data['lyrics']:
#             for lyric in section_data['lyrics']:
#                 events.append({
#                     'time': lyric['relative_time'],
#                     'type': 'lyric',
#                     'content': lyric['text']
#                 })
#
#         # Add instrument entrance events (simplified)
#         for instrument in section_data['active_instruments']:
#             events.append({
#                 'time': 0,  # Would need more analysis to determine exact entrance
#                 'type': 'instrument_active',
#                 'content': instrument['description']
#             })
#
#         return sorted(events, key=lambda x: x['time'])
#
#
# # Usage example
# if __name__ == "__main__":
#     # Initialize with your YAML file
#     extractor = MusicDataExtractor('paste.txt')  # Your YAML file
#
#     # Extract Verse 1 data
#     verse1_data = extractor.extract_section_data("Verse 1")
#
#     # Create training sample
#     training_sample = extractor.create_training_sample(verse1_data)
#
#     # Save in multiple formats
#     output_dir = extractor.save_training_data(training_sample)
#
#     print("\n=== VERSE 1 TRAINING SAMPLE ===")
#     print(f"Duration: {training_sample['section']['duration']:.2f} seconds")
#     print(f"Active instruments: {len(training_sample['instrumentation'])}")
#     print(f"Lyric lines: {len(training_sample['lyrics']) if training_sample['lyrics'] else 0}")
#     print(f"Audio files extracted: {len(training_sample['audio_files'])}")
#     print(f"Training data saved to: {output_dir}/")
#
#     # Load the temporal parquet to see the structure
#     temporal_df = pd.read_parquet(f"{output_dir}/verse1_temporal.parquet")
#     print(f"\nTemporal DataFrame shape: {temporal_df.shape}")
#     print(temporal_df.head())

import os
import app.objects.rainbow_song_meta
import app.objects.rainbow_song

if __name__ == "__main__":
    meta = app.objects.rainbow_song_meta.RainbowSongMeta(yaml_file_name="01_01.yml", base_path=os.path.join(os.path.dirname(__file__), "staged_raw_material"), track_materials_path="01_01")
    songs = app.objects.rainbow_song.RainbowSong(meta_data=meta, extracts=None)
