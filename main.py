
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
    songs.create_dataframe()


