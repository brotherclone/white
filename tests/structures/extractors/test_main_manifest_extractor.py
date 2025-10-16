import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from app.structures.extractors.main_manifest_extractor import ManifestExtractor
from app.structures.enums.player import RainbowPlayer


@pytest.fixture
def mock_manifest_with_players():
    """Mock manifest with multiple audio tracks and players"""
    return {
        'manifest_id': 'test_01',
        'title': 'Test Song',
        'bpm': 120,
        'tempo': '4/4',
        'key': 'C',
        'rainbow_color': 'RED',
        'concept': 'Test concept',
        'mood': ['energetic'],
        'structure': [
            {
                'section_name': 'Intro',
                'start_time': '[00:00.000]',
                'end_time': '[00:30.000]',
                'description': 'Opening section'
            },
            {
                'section_name': 'Verse',
                'start_time': '[00:30.000]',
                'end_time': '[01:00.000]',
                'description': 'First verse'
            }
        ],
        'audio_tracks': [
            {
                'id': 1,
                'description': 'Lead Vocals',
                'audio_file': 'vocals.wav',
                # No player - should default to GABE
            },
            {
                'id': 2,
                'description': 'Background Vocals',
                'audio_file': 'bg_vox.wav',
                'player': 'REMEZ'
            },
            {
                'id': 3,
                'description': 'Guitar',
                'audio_file': 'guitar.wav',
                'player': 'JOSH'
            }
        ]
    }


@pytest.fixture
def temp_manifest_with_audio_files(mock_manifest_with_players):
    """Create temporary manifest with audio files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write manifest
        import yaml
        manifest_path = os.path.join(tmpdir, 'test_01.yml')
        with open(manifest_path, 'w') as f:
            yaml.dump(mock_manifest_with_players, f)

        # Create dummy audio files
        for track in mock_manifest_with_players['audio_tracks']:
            audio_path = os.path.join(tmpdir, track['audio_file'])
            Path(audio_path).touch()

        yield manifest_path, tmpdir


class TestManifestExtractorWithPlayers:
    """Tests for ManifestExtractor with player attribution"""

    @patch('app.util.manifest_loader.load_manifest')
    def test_get_player_for_track_defaults_to_gabe(self, mock_load_manifest):
        """Test that tracks without player default to GABE"""
        extractor = ManifestExtractor(manifest_id='test_01', load_manifest_on_init=False)

        # Track without player field
        track_no_player = Mock()
        track_no_player.player = None

        player = extractor._get_player_for_track(track_no_player)
        assert player == RainbowPlayer.GABE.value

    @patch('app.util.manifest_loader.load_manifest')
    def test_get_player_for_track_returns_specified_player(self, mock_load_manifest):
        """Test that tracks with player return correct value"""
        extractor = ManifestExtractor(manifest_id='test_01', load_manifest_on_init=False)

        # Track with player
        track_with_player = Mock()
        track_with_player.player = 'REMEZ'

        player = extractor._get_player_for_track(track_with_player)
        assert player == 'REMEZ'

    @patch('app.util.manifest_loader.load_manifest')
    @patch.object(ManifestExtractor, 'load_audio_segment')
    def test_generate_multimodal_segments_with_multiple_tracks(
            self, mock_load_audio, mock_load_manifest, mock_manifest_with_players
    ):
        """Test that all audio tracks are loaded with player attribution"""
        # Setup
        mock_load_manifest.return_value = Mock(**mock_manifest_with_players)
        mock_load_audio.return_value = {
            'rms_energy': 0.5,
            'spectral_centroid': 2000.0,
            'attack_time': 0.05,
            'decay_profile': [0.9, 0.7, 0.5]
        }

        extractor = ManifestExtractor(manifest_id='test_01')

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create manifest file
            import yaml
            manifest_path = os.path.join(tmpdir, 'test_01.yml')
            with open(manifest_path, 'w') as f:
                yaml.dump(mock_manifest_with_players, f)

            # Create audio files
            for track in mock_manifest_with_players['audio_tracks']:
                Path(os.path.join(tmpdir, track['audio_file'])).touch()

            # Generate segments
            df = extractor.generate_multimodal_segments(manifest_path)

            # Assertions
            assert len(df) == 2  # Two sections
            assert 'audio_tracks_features' in df.columns
            assert 'players' in df.columns
            assert 'player_count' in df.columns

            # Check first segment has all tracks
            first_segment_tracks = df.iloc[0]['audio_tracks_features']
            assert len(first_segment_tracks) == 3

            # Verify player attribution
            players = [t['player'] for t in first_segment_tracks]
            assert RainbowPlayer.GABE.value in players  # Default for track 1
            assert 'REMEZ' in players
            assert 'JOSH' in players

            # Verify track metadata preserved
            track_descriptions = [t['description'] for t in first_segment_tracks]
            assert 'Lead Vocals' in track_descriptions
            assert 'Background Vocals' in track_descriptions
            assert 'Guitar' in track_descriptions

    @patch('app.util.manifest_loader.load_manifest')
    def test_flatten_and_unflatten_preserves_players(self, mock_load_manifest):
        """Test that player info survives parquet round-trip"""
        # Create a DataFrame with audio tracks
        df = pd.DataFrame([{
            'manifest_id': 'test_01',
            'segment_id': 'test_01_intro',
            'audio_tracks_features': [
                {
                    'track_id': 1,
                    'description': 'Vocals',
                    'player': RainbowPlayer.GABE.value,
                    'rms_energy': 0.5,
                    'spectral_centroid': 2000.0
                },
                {
                    'track_id': 2,
                    'description': 'BG Vox',
                    'player': 'REMEZ',
                    'rms_energy': 0.3,
                    'spectral_centroid': 1500.0
                }
            ],
            'lyrical_content': []
        }])

        extractor = ManifestExtractor(manifest_id='test_01', load_manifest_on_init=False)

        # Flatten
        df_flat = extractor._flatten_for_parquet(df)

        # Check flattened columns exist
        assert 'audio_tracks_json' in df_flat.columns
        assert 'players_csv' in df_flat.columns

        # Check player CSV
        assert RainbowPlayer.GABE.value in df_flat.iloc[0]['players_csv']
        assert 'REMEZ' in df_flat.iloc[0]['players_csv']

        # Unflatten
        df_restored = extractor._unflatten_from_parquet(df_flat)

        # Verify restoration
        assert 'audio_tracks_features' in df_restored.columns
        restored_tracks = df_restored.iloc[0]['audio_tracks_features']
        assert len(restored_tracks) == 2
        assert restored_tracks[0]['player'] == RainbowPlayer.GABE.value
        assert restored_tracks[1]['player'] == 'REMEZ'

    @patch('app.util.manifest_loader.load_manifest')
    def test_calculate_boundary_fluidity_with_multiple_tracks(self, mock_load_manifest):
        """Test boundary fluidity calculation with multiple audio tracks"""
        segment = {
            'lyrical_content': [],
            'audio_tracks_features': [
                {
                    'player': RainbowPlayer.GABE.value,
                    'attack_time': 0.05,  # Fast attack
                    'decay_profile': [0.9, 0.7, 0.5, 0.3]
                },
                {
                    'player': 'REMEZ',
                    'attack_time': 0.15,  # Slower attack
                    'decay_profile': [0.95, 0.8]
                }
            ],
            'midi_features': {}
        }

        extractor = ManifestExtractor(manifest_id='test_01', load_manifest_on_init=False)
        score = extractor._calculate_boundary_fluidity_score(segment)

        # Should have some score from audio features
        assert score > 0
        assert score <= 1.0

    @patch('app.util.manifest_loader.load_manifest')
    def test_deprecated_audio_path_parameter_warning(self, mock_load_manifest, capsys):
        """Test that passing mms_audio_path shows deprecation warning"""
        mock_manifest = Mock(
            manifest_id='test_01',
            title='Test',
            structure=[],
            audio_tracks=[]
        )
        mock_load_manifest.return_value = mock_manifest

        extractor = ManifestExtractor(manifest_id='test_01')

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, 'test.yml')
            Path(manifest_path).touch()

            # Call with deprecated parameter
            extractor.generate_multimodal_segments(
                manifest_path,
                mms_audio_path='some_file.wav'  # Should trigger warning
            )

            captured = capsys.readouterr()
            assert 'WARNING' in captured.out
            assert 'deprecated' in captured.out.lower()


class TestPlayerEnumIntegration:
    """Test integration with RainbowPlayer enum"""

    def test_all_players_are_valid(self):
        """Test that all RainbowPlayer enum values are valid"""
        from app.structures.enums.player import RainbowPlayer

        expected_players = ['GABE', 'JOSH', 'REMEZ', 'MARVIN', 'GRAHAM']

        for player in expected_players:
            assert hasattr(RainbowPlayer, player)
            enum_member = getattr(RainbowPlayer, player)
            assert enum_member.value is not None

    @patch('app.util.manifest_loader.load_manifest')
    def test_enum_value_extraction(self, mock_load_manifest):
        """Test extracting string value from enum"""
        extractor = ManifestExtractor(manifest_id='test_01', load_manifest_on_init=False)

        # Test with enum instance
        track_with_enum = Mock()
        track_with_enum.player = RainbowPlayer.JOSH

        player = extractor._get_player_for_track(track_with_enum)
        assert player == RainbowPlayer.JOSH.value
        assert player == "Josh Plotner"