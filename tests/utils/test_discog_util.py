import unittest

from app.utils.discog_util import get_discogs_artist, search_discogs_artist, get_discogs_group_members, get_discogs_release_list

class TestDiscogUtil(unittest.IsolatedAsyncioTestCase):
    async def test_get_discogs_artist(self):
        artist_id = 1  # Replace with a valid artist ID for testing
        artist = await get_discogs_artist(artist_id)
        self.assertIsNotNone(artist)
        self.assertEqual(artist.id, artist_id)

    async def test_search_discogs_artist(self):
        artist_name = "The Beatles"  # Replace with a valid artist name for testing
        artist = await search_discogs_artist(artist_name)
        self.assertIsNotNone(artist)
        self.assertIn(artist_name, artist.name)

    async def test_get_discogs_group_members(self):
        group_name = "The Beatles"  # Replace with a valid group name for testing
        members = await get_discogs_group_members(group_name)
        self.assertGreater(len(members), 0)

    async def test_get_discogs_release_list(self):
        artist_id = 1  # Replace with a valid artist ID for testing
        releases = await get_discogs_release_list(artist_id, per_page=5, page=1, sort='year', sort_order='asc')
        self.assertGreater(len(releases), 0)