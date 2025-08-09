import unittest
import os
import datetime
import tempfile
from unittest.mock import patch, MagicMock

from app.objects.rainbow_song import RainbowSong
from app.objects.rainbow_song_meta import RainbowSongMeta
from app.objects.multimodal_extract import MultimodalExtract, MultimodalExtractModel


class TestRainbowSong(unittest.TestCase):
    """Test suite for the RainbowSong class."""

    @patch('app.objects.rainbow_song.mido')
    @patch('app.objects.rainbow_song.AudioSegment')
    def setUp(self, mock_audio_segment, mock_mido):
        """Set up test fixtures before each test method."""
        # Create a mock for RainbowSongMeta
        self.mock_meta = MagicMock(spec=RainbowSongMeta)
        self.mock_meta.data.structure = []
        self.mock_meta.data.main_audio_file = "test.wav"
        self.mock_meta.data.audio_tracks = []
        self.mock_meta.data.lrc_file = "test.lrc"
        self.mock_meta.data.bpm = 120
        self.mock_meta.data.key = "C major"
        self.mock_meta.data.title = "Test Song"
        self.mock_meta.data.artist = "Test Artist"
        self.mock_meta.data.album_title = "Test Album"
        self.mock_meta.data.release_date = "2025-01-01"
        self.mock_meta.data.album_sequence = 1
        self.mock_meta.data.TRT = datetime.timedelta(minutes=3)
        self.mock_meta.data.vocals = True
        self.mock_meta.data.lyrics = True
        self.mock_meta.data.mood = ["Happy"]
        self.mock_meta.data.genres = ["Pop"]
        self.mock_meta.data.sounds_like = []
        self.mock_meta.data.rainbow_color = None
        self.mock_meta.data.concept = None
        self.mock_meta.data.reference_plans_paths = []

        # Set up base path and track materials path
        self.mock_meta.base_path = "/test/path"
        self.mock_meta.track_materials_path = "test_track"

        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()

        # Set up mock AudioSegment behavior
        mock_segment = MagicMock()
        mock_segment.__getitem__.return_value = mock_segment
        mock_segment.set_frame_rate.return_value = mock_segment
        mock_audio_segment.from_file.return_value = mock_segment

        # Create the RainbowSong instance with the mock meta
        with patch('app.objects.rainbow_song.os.path.join', return_value=self.temp_dir.name):
            with patch('app.objects.rainbow_song.has_significant_audio', return_value=True):
                with patch('app.objects.rainbow_song.RainbowSong._segment_song'):
                    self.song = RainbowSong(meta_data=self.mock_meta)
                    # Manually set extracts to avoid calling _segment_song
                    self.song.extracts = []

    def tearDown(self):
        """Clean up after each test method."""
        self.temp_dir.cleanup()

    def test_segment_song_with_short_section(self):
        """Test segmenting a song with a section shorter than the maximum duration."""
        # Create a mock section that's shorter than MAXIMUM_EXTRACT_DURATION
        mock_section = MagicMock()
        mock_section.duration = datetime.timedelta(seconds=10)
        mock_section.start_time = "00:00"
        mock_section.end_time = "00:10"
        mock_section.section_name = "Short Section"
        mock_section.section_description = "A short test section"
        mock_section.midi_group = None

        self.mock_meta.data.structure = [mock_section]

        # Test the segmentation
        with patch('app.objects.rainbow_song.lrc_to_seconds', side_effect=lambda x: datetime.timedelta(seconds=int(x.split(':')[1]))):
            with patch('app.objects.rainbow_song.RainbowSong._process_all_extracts'):
                self.song._segment_song()

        # Check that one extract was created
        self.assertEqual(len(self.song.extracts), 1)
        self.assertEqual(self.song.extracts[0].extract_data.section_name, "Short Section")

    def test_segment_song_with_long_section(self):
        """Test segmenting a song with a section longer than the maximum duration."""
        # Create a mock section that's longer than MAXIMUM_EXTRACT_DURATION
        mock_section = MagicMock()
        mock_section.duration = datetime.timedelta(seconds=60)  # 60 seconds, should be split
        mock_section.start_time = "00:00"
        mock_section.end_time = "01:00"
        mock_section.section_name = "Long Section"
        mock_section.section_description = "A long test section"
        mock_section.midi_group = None

        self.mock_meta.data.structure = [mock_section]

        # Test the segmentation
        with patch('app.objects.rainbow_song.lrc_to_seconds', side_effect=lambda x: datetime.timedelta(seconds=int(x.split(':')[1]))):
            with patch('app.objects.rainbow_song.RainbowSong._process_all_extracts'):
                self.song._segment_song()

        # Check that multiple extracts were created (60s / 29s = 3 segments)
        self.assertGreater(len(self.song.extracts), 1)
        # Check that the first extract has the correct name format
        self.assertTrue("Long Section (part 1/" in self.song.extracts[0].extract_data.section_name)

    @patch('app.objects.rainbow_song.open')
    def test_extract_lyrics(self, mock_open):
        """Test extracting lyrics from an LRC file."""
        # Create a mock extract
        extract = MultimodalExtract(
            extract_data=MultimodalExtractModel(
                start_time=datetime.timedelta(seconds=10),
                end_time=datetime.timedelta(seconds=20),
                duration=datetime.timedelta(seconds=10),
                section_name="Test Section",
                section_description="Test Description",
                sequence=1,
                events=[],
                extract_lrc=None,
                extract_lyrics=None,
                midi_group=None
            )
        )

        # Mock the LRC file content
        mock_file = MagicMock()
        mock_file.__enter__.return_value.readlines.return_value = [
            "[ti:Test Song]\n",
            "[00:05]Line before extract\n",
            "[00:15]Line within extract\n",
            "[00:25]Line after extract\n"
        ]
        mock_open.return_value = mock_file

        # Mock lrc_to_seconds to return expected values
        with patch('app.objects.rainbow_song.lrc_to_seconds', side_effect=[
            datetime.timedelta(seconds=5),
            datetime.timedelta(seconds=15),
            datetime.timedelta(seconds=25)
        ]):
            self.song.extract_lyrics(extract)

        # Check that only lyrics within the extract's time range were included
        self.assertIn("Line within extract", extract.extract_data.extract_lyrics or "")
        self.assertNotIn("Line before extract", extract.extract_data.extract_lyrics or "")
        self.assertNotIn("Line after extract", extract.extract_data.extract_lyrics or "")

    @patch('app.objects.rainbow_song.os.path.join')
    @patch('app.objects.rainbow_song.RainbowSong._extract_audio_segment')
    def test_extract_audio(self, mock_extract_segment, mock_join):
        """Test extracting audio segments."""
        # Create a mock extract
        extract = MultimodalExtract(
            extract_data=MultimodalExtractModel(
                start_time=datetime.timedelta(seconds=10),
                end_time=datetime.timedelta(seconds=20),
                duration=datetime.timedelta(seconds=10),
                section_name="Test Section",
                section_description="Test Description",
                sequence=1,
                events=[],
                extract_lrc=None,
                extract_lyrics=None,
                midi_group=None
            )
        )

        # Set up mock to return a valid audio segment
        mock_segment = MagicMock()
        mock_extract_segment.return_value = mock_segment

        # Mock os.path.join to return a valid path
        mock_join.return_value = "/test/path/test.wav"

        # Set up the track in the metadata
        self.mock_meta.data.main_audio_file = "test.wav"

        # Test the audio extraction with mocked export
        with patch.object(mock_segment, 'export') as mock_export:
            self.song.extract_audio(extract)

        # Check that the export method was called
        mock_export.assert_called_once()

        # Check that an event was added to the extract
        self.assertEqual(len(extract.extract_data.events), 1)

    @patch('app.objects.rainbow_song.os.path.join')
    @patch('app.objects.rainbow_song.mido.MidiFile')
    @patch('app.objects.rainbow_song.split_midi_file_by_segment')
    @patch('app.objects.rainbow_song.midi_to_bytes')
    def test_extract_midi(self, mock_midi_to_bytes, mock_split, mock_midi_file, mock_join):
        """Test extracting MIDI segments."""
        # Create a mock extract
        extract = MultimodalExtract(
            extract_data=MultimodalExtractModel(
                start_time=datetime.timedelta(seconds=10),
                end_time=datetime.timedelta(seconds=20),
                duration=datetime.timedelta(seconds=10),
                section_name="Test Section",
                section_description="Test Description",
                sequence=1,
                events=[],
                extract_lrc=None,
                extract_lyrics=None,
                midi_group=None
            )
        )

        # Set up mock audio track with MIDI file
        mock_track = MagicMock()
        mock_track.midi_file = "test.mid"
        mock_track.midi_group = None
        mock_track.midi_group_file = None
        self.mock_meta.data.audio_tracks = [mock_track]

        # Set up mock MIDI file
        mock_midi = MagicMock()
        mock_midi.ticks_per_beat = 480
        mock_track = MagicMock()
        mock_track.name = "Piano"

        # Create a note-on message within the extract's time range
        note_on_msg = MagicMock()
        note_on_msg.type = "note_on"
        note_on_msg.velocity = 64
        note_on_msg.note = 60
        note_on_msg.time = 100

        mock_track.iter_events.return_value = [note_on_msg]
        mock_midi.tracks = [mock_track]
        mock_midi_file.return_value = mock_midi

        # Mock os.path.join to return a valid path
        mock_join.return_value = "/test/path/test.mid"

        # Mock split_midi_file_by_segment to return a path
        mock_split.return_value = "/test/output/segment.mid"

        # Mock midi_to_bytes to return some bytes
        mock_midi_to_bytes.return_value = b'midi_data'

        # Mock tick2second to convert ticks to seconds
        with patch('app.objects.rainbow_song.mido.tick2second', side_effect=lambda t, tpb, tempo: t/tpb):
            # Call extract_midi
            self.song.extract_midi(extract)

        # Check that an event was added to the extract
        self.assertEqual(len(extract.extract_data.events), 1)

    def test_create_training_samples(self):
        """Test creating training samples from extracts."""
        # Create a mock extract
        extract = MultimodalExtract(
            extract_data=MultimodalExtractModel(
                start_time=datetime.timedelta(seconds=10),
                end_time=datetime.timedelta(seconds=20),
                duration=datetime.timedelta(seconds=10),
                section_name="Test Section",
                section_description="Test Description",
                sequence=1,
                events=[],
                extract_lrc="Test LRC",
                extract_lyrics="Test Lyrics",
                midi_group=None
            )
        )

        # Add a mock event to the extract
        mock_event = MagicMock()
        mock_event.type = "mix_audio"
        mock_event.content = {
            'file_name': 'test.wav',
            'description': 'Stereo Mix',
            'id': 'mix',
            'source_audio_file': 'test.wav'
        }
        extract.extract_data.events.append(mock_event)

        # Add the extract to the song
        self.song.extracts = [extract]

        # Mock the TrainingSampleValidator
        with patch('app.objects.rainbow_song.TrainingSampleValidator') as mock_validator_class:
            mock_validator = MagicMock()
            mock_validator_class.return_value = mock_validator

            # Mock validation_summary
            mock_validation_summary = MagicMock()
            mock_validation_summary.samples_with_errors = 0
            mock_validator.validate_dataframe.return_value = mock_validation_summary

            # Mock DataFrame.to_parquet
            with patch('app.objects.rainbow_song.pd.DataFrame.to_parquet') as mock_to_parquet:
                # Mock audio_to_byes
                with patch('app.objects.rainbow_song.audio_to_byes', return_value=b'audio_data'):
                    # Call create_training_samples
                    self.song.create_training_samples()

        # Check that training samples were created
        self.assertEqual(len(self.song.training_samples), 1)

        # Check that to_parquet was called
        mock_to_parquet.assert_called_once()


if __name__ == '__main__':
    unittest.main()
