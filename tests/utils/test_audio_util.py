import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from app.utils.audio_util import has_significant_audio, compute_rms, get_microseconds_per_beat


class TestHasSignificantAudio(unittest.TestCase):

    def test_empty_audio_chunk(self):
        # Test with an empty audio chunk
        audio_chunk = MagicMock()
        audio_chunk.__len__.return_value = 0

        self.assertFalse(has_significant_audio(audio_chunk))

    def test_empty_samples(self):
        # Test with an audio chunk that returns empty samples
        audio_chunk = MagicMock()
        audio_chunk.__len__.return_value = 1
        audio_chunk.get_array_of_samples.return_value = []

        self.assertFalse(has_significant_audio(audio_chunk))

    def test_low_volume_audio(self):
        # Test with samples that are below the threshold
        audio_chunk = MagicMock()
        audio_chunk.__len__.return_value = 1
        # Create samples with low amplitude
        audio_chunk.get_array_of_samples.return_value = np.array([100, 150, 120, 80])

        self.assertFalse(has_significant_audio(audio_chunk, threshold_db=-20))

    def test_high_volume_audio(self):
        # Test with samples that are above the threshold
        audio_chunk = MagicMock()
        audio_chunk.__len__.return_value = 1
        # Create samples with high amplitude
        audio_chunk.get_array_of_samples.return_value = np.array([10000, 12000, 11000, 9500])

        self.assertTrue(has_significant_audio(audio_chunk))

    def test_custom_threshold(self):
        # Test with a custom threshold value
        audio_chunk = MagicMock()
        audio_chunk.__len__.return_value = 1
        audio_chunk.get_array_of_samples.return_value = np.array([5000, 4800, 5200, 4900])

        self.assertTrue(has_significant_audio(audio_chunk, threshold_db=-40))
        self.assertFalse(has_significant_audio(audio_chunk, threshold_db=-10))


class TestComputeRMS(unittest.TestCase):

    def test_normal_case(self):
        """Test RMS calculation with normal audio samples."""
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        expected_rms = np.sqrt(np.mean(np.array(samples) ** 2))
        self.assertAlmostEqual(compute_rms(samples), expected_rms)

    def test_empty_array(self):
        """Test with an empty array."""
        with patch('builtins.print') as mock_print:
            result = compute_rms([])
            self.assertEqual(result, 0.0)
            mock_print.assert_called_once()

    def test_nan_and_inf_values(self):
        """Test with array containing NaN and Inf values."""
        samples = [1.0, 2.0, np.nan, 4.0, np.inf]
        # Only 1.0, 2.0, and 4.0 should be used in calculation
        valid_samples = np.array([1.0, 2.0, 4.0])
        expected_rms = np.sqrt(np.mean(valid_samples ** 2))
        self.assertAlmostEqual(compute_rms(samples), expected_rms)

    def test_all_nan_inf(self):
        """Test with array containing only NaN and Inf values."""
        with patch('builtins.print') as mock_print:
            result = compute_rms([np.nan, np.inf, -np.inf])
            self.assertEqual(result, 0.0)
            mock_print.assert_called_once()

    def test_negative_values(self):
        """Test with negative values."""
        samples = [-1.0, -2.0, -3.0]
        expected_rms = np.sqrt(np.mean(np.array(samples) ** 2))
        self.assertAlmostEqual(compute_rms(samples), expected_rms)

    def test_zero_values(self):
        """Test with all zeros."""
        samples = [0, 0, 0]
        self.assertEqual(compute_rms(samples), 0.0)


class TestGetMicroSecondsPerBeat(unittest.TestCase):

    def test_normal_case(self):
        self.assertEqual(get_microseconds_per_beat(120), 500000.0)  # 120 BPM = 500,000 microseconds per beat
        self.assertEqual(get_microseconds_per_beat(60), 1000000.0)  # 60 BPM = 1,000,000 microseconds per beat

    def test_floating_point_bpm(self):
        self.assertAlmostEqual(get_microseconds_per_beat(123.45), 486027.54, delta=1.0)
        self.assertEqual(get_microseconds_per_beat(1), 60000000.0)
        self.assertEqual(get_microseconds_per_beat(300), 200000.0)

    def test_invalid_bpm(self):
        with self.assertRaises(ValueError):
            get_microseconds_per_beat(0)  # Zero BPM should raise ValueError

        with self.assertRaises(ValueError):
            get_microseconds_per_beat(-120)  # Negative BPM should raise ValueError


if __name__ == '__main__':
    unittest.main()
