import unittest

from app.utils.string_util import safe_filename, just_lyrics, make_lrc_fragment, to_str_dict, bytes_to_base64_str, get_random_musical_key, uuid_representer, enum_representer, convert_to_rainbow_color

class TestStringUtil(unittest.TestCase):
    def test_safe_filename(self):
        self.assertEqual(safe_filename("Test File.txt"), "Test_File.txt")
        self.assertEqual(safe_filename("Invalid/Name*?<>|"), "Invalid_Name____")
        self.assertEqual(safe_filename("  Leading and trailing spaces  "), "Leading_and_trailing_spaces")

    def test_just_lyrics(self):
        # Placeholder for actual implementation
        self.assertIsNone(just_lyrics(None))

    def test_make_lrc_fragment(self):
        # Placeholder for actual implementation
        self.assertIsNone(make_lrc_fragment("Album", "Song", "Artist", None))

    def test_to_str_dict(self):
        d = {'key1': 'value1', 'key2': None, 'key3': True, 'key4': b'bytes'}
        expected = {'key1': 'value1', 'key2': None, 'key3': 'True', 'key4': b'bytes'}
        self.assertEqual(to_str_dict(d), expected)

    def test_bytes_to_base64_str(self):
        self.assertEqual(bytes_to_base64_str(b'test'), 'dGVzdA==')
        self.assertIsNone(bytes_to_base64_str(None))

    def test_get_random_musical_key(self):
        key = get_random_musical_key()
        self.assertIn(key.split()[0], ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
        self.assertIn(key.split()[1], ['major', 'minor', 'dorian', 'phrygian', 'lydian', 'mixolydian', 'locrian'])

    def test_uuid_representer(self):
        from yaml import Dumper
        class MockDumper(Dumper):
            def represent_scalar(self, tag, value):
                return f"{tag}: {value}"

        dumper = MockDumper()
        result = uuid_representer(dumper, "123e4567-e89b-12d3-a456-426614174000")
        self.assertEqual(result, "tag:yaml.org,2002:str: 123e4567-e89b-12d3-a456-426614174000")

    def test_enum_representer(self):
        from yaml import Dumper