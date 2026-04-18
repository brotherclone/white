"""Tests for MusicExtractor — narrative-to-song-proposal extraction."""

import random

from app.structures.concepts.pulsar_palace_room import PulsarPalaceRoom
from app.structures.generators.music_extractor import MusicExtractor
from app.structures.manifests.song_proposal import SongProposalIteration


def _make_room(
    name="The Crystal Chamber",
    room_id="room_001",
    atmosphere="elegant - ornate baroque grandeur",
    room_type="ballroom",
    description="A vast elegant chamber with baroque ornamentation.",
    inhabitants=None,
):
    return PulsarPalaceRoom(
        name=name,
        room_id=room_id,
        atmosphere=atmosphere,
        room_type=room_type,
        description=description,
        inhabitants=inhabitants or [],
    )


# ---------------------------------------------------------------------------
# _extract_mood
# ---------------------------------------------------------------------------


class TestExtractMood:
    def setup_method(self):
        self.extractor = MusicExtractor()

    def test_keyword_in_description_triggers_mood(self):
        room = _make_room(description="A baroque hall with crystalline walls.")
        moods = self.extractor._extract_mood(room, "silence fills the space")
        assert "baroque" in moods or "ornate" in moods or "classical" in moods

    def test_keyword_in_narrative_triggers_mood(self):
        room = _make_room(description="Plain room.")
        moods = self.extractor._extract_mood(room, "the crystalline structures glow")
        assert "bright" in moods or "shimmering" in moods or "ethereal" in moods

    def test_mysterious_always_present(self):
        room = _make_room(description="A plain room.", atmosphere="neutral - plain")
        moods = self.extractor._extract_mood(room, "nothing remarkable happens")
        assert "mysterious" in moods

    def test_party_and_dance_adds_playful(self):
        # Use a neutral atmosphere to avoid keyword matches that fill up the 5-item cap
        room = _make_room(description="A dance floor.", atmosphere="neutral - plain")
        moods = self.extractor._extract_mood(
            room, "the party starts and everyone dance"
        )
        assert "playful" in moods

    def test_scream_adds_terrifying(self):
        # Use a neutral atmosphere to avoid keyword matches that fill up the 5-item cap
        room = _make_room(description="A plain room.", atmosphere="neutral - plain")
        moods = self.extractor._extract_mood(
            room, "something causes everyone to scream"
        )
        assert "terrifying" in moods

    def test_terror_also_adds_terrifying(self):
        room = _make_room(description="A plain room.", atmosphere="neutral - plain")
        moods = self.extractor._extract_mood(room, "pure terror grips the party")
        assert "terrifying" in moods

    def test_weep_adds_melancholic(self):
        room = _make_room(description="A plain room.", atmosphere="neutral - plain")
        moods = self.extractor._extract_mood(room, "the character begins to weep")
        assert "melancholic" in moods

    def test_result_capped_at_five(self):
        room = _make_room(
            description="baroque crystalline psychedelic threatening pulsing cosmic warped"
        )
        moods = self.extractor._extract_mood(room, "")
        assert len(moods) <= 5

    def test_returns_list(self):
        room = _make_room()
        result = self.extractor._extract_mood(room, "")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _calculate_bpm
# ---------------------------------------------------------------------------


class TestCalculateBpm:
    def setup_method(self):
        self.extractor = MusicExtractor()
        random.seed(42)

    def test_result_within_midi_range(self):
        bpm = self.extractor._calculate_bpm("elegant", "a quiet gentle scene")
        assert 40 <= bpm <= 200

    def test_tension_keywords_push_bpm_up(self):
        random.seed(0)
        bpm_calm = self.extractor._calculate_bpm(
            "elegant", "still quiet gentle slow frozen"
        )
        random.seed(0)
        bpm_tense = self.extractor._calculate_bpm(
            "elegant", "violent chaos scream run attack panic"
        )
        assert bpm_tense > bpm_calm

    def test_calm_keywords_push_bpm_down(self):
        random.seed(0)
        bpm_neutral = self.extractor._calculate_bpm("weird", "")
        random.seed(0)
        bpm_calm = self.extractor._calculate_bpm(
            "weird", "still frozen quiet gentle slow"
        )
        assert bpm_calm <= bpm_neutral

    def test_unknown_atmosphere_uses_default_range(self):
        bpm = self.extractor._calculate_bpm("unknown_type", "")
        assert 40 <= bpm <= 200

    def test_returns_int(self):
        bpm = self.extractor._calculate_bpm("elegant", "")
        assert isinstance(bpm, int)

    def test_known_atmospheres(self):
        for atm in ("elegant", "surreal", "weird", "dangerous"):
            bpm = self.extractor._calculate_bpm(atm, "")
            assert 40 <= bpm <= 200


# ---------------------------------------------------------------------------
# _determine_key
# ---------------------------------------------------------------------------


class TestDetermineKey:
    def setup_method(self):
        self.extractor = MusicExtractor()
        random.seed(42)

    def test_result_is_string(self):
        key = self.extractor._determine_key("elegant", ["mysterious"])
        assert isinstance(key, str)
        assert len(key) > 0

    def test_dark_mood_prefers_minor(self):
        # Run many times; with "dark" in mood the result should always be minor
        random.seed(7)
        keys = set()
        for _ in range(20):
            key = self.extractor._determine_key("surreal", ["dark", "ominous"])
            keys.add(key)
        assert all("minor" in k for k in keys)

    def test_no_dark_mood_may_return_any_key(self):
        random.seed(1)
        key = self.extractor._determine_key("elegant", ["mysterious"])
        assert isinstance(key, str)

    def test_unknown_atmosphere_uses_fallback(self):
        key = self.extractor._determine_key("unknown", [])
        assert key == "C major"

    def test_all_known_atmospheres(self):
        for atm in ("elegant", "surreal", "weird", "dangerous"):
            key = self.extractor._determine_key(atm, [])
            assert isinstance(key, str)


# ---------------------------------------------------------------------------
# _determine_genres
# ---------------------------------------------------------------------------


class TestDetermineGenres:
    def setup_method(self):
        self.extractor = MusicExtractor()
        random.seed(42)

    def test_base_genres_always_included(self):
        genres = self.extractor._determine_genres("elegant", [])
        for base in ("electronic", "ambient", "experimental", "kosmische"):
            assert base in genres

    def test_result_capped_at_six(self):
        genres = self.extractor._determine_genres(
            "elegant", ["baroque", "playful", "terrifying", "dark"]
        )
        assert len(genres) <= 6

    def test_returns_list(self):
        genres = self.extractor._determine_genres("surreal", [])
        assert isinstance(genres, list)

    def test_baroque_mood_adds_baroque_pop(self):
        genres = self.extractor._determine_genres("elegant", ["baroque"] * 10)
        assert "baroque pop" in genres

    def test_playful_mood_adds_novelty(self):
        genres = self.extractor._determine_genres("unknown", ["playful"])
        assert "novelty" in genres

    def test_dark_mood_adds_dark_ambient(self):
        # Use unknown atmosphere (no atmosphere genres) so total stays under cap
        genres = self.extractor._determine_genres("unknown", ["dark"])
        assert "dark ambient" in genres

    def test_no_duplicates(self):
        genres = self.extractor._determine_genres("elegant", ["baroque", "playful"])
        assert len(genres) == len(set(genres))

    def test_unknown_atmosphere_uses_base_only(self):
        random.seed(42)
        genres = self.extractor._determine_genres("unknown", [])
        for base in ("electronic", "ambient", "experimental", "kosmische"):
            assert base in genres


# ---------------------------------------------------------------------------
# _generate_iteration_id
# ---------------------------------------------------------------------------


class TestGenerateIterationId:
    def setup_method(self):
        self.extractor = MusicExtractor()

    def test_basic_format(self):
        # The regex extracts the trailing digits as-is, preserving leading zeros
        result = self.extractor._generate_iteration_id("Crystal Chamber", "room_003")
        assert result == "pulsar_palace_crystal_chamber_v003"

    def test_spaces_become_underscores(self):
        result = self.extractor._generate_iteration_id("The Grand Hall", "room_1")
        assert " " not in result

    def test_hyphens_become_underscores(self):
        result = self.extractor._generate_iteration_id("Anti-Chamber", "room_1")
        assert "-" not in result

    def test_special_chars_removed(self):
        result = self.extractor._generate_iteration_id("Room! #1", "room_1")
        assert "!" not in result
        assert "#" not in result

    def test_no_trailing_number_uses_1(self):
        result = self.extractor._generate_iteration_id("Test Room", "roomX")
        assert result.endswith("_v1")

    def test_version_extracted_from_id(self):
        result = self.extractor._generate_iteration_id("Test Room", "room_42")
        assert result.endswith("_v42")

    def test_starts_with_pulsar_palace(self):
        result = self.extractor._generate_iteration_id("Void", "room_1")
        assert result.startswith("pulsar_palace_")


# ---------------------------------------------------------------------------
# _extract_key_phrase
# ---------------------------------------------------------------------------


class TestExtractKeyPhrase:
    def setup_method(self):
        self.extractor = MusicExtractor()

    def test_finds_reality_phrase(self):
        narrative = "Everything changes. Reality bends and warps."
        result = self.extractor._extract_key_phrase(narrative)
        assert "Reality bends and warps." in result or result != ""

    def test_finds_pulsar_phrase(self):
        narrative = "The pulsar emits a wave of energy."
        result = self.extractor._extract_key_phrase(narrative)
        assert "pulsar" in result.lower()

    def test_finds_time_phrase(self):
        narrative = "Time stands still in this place."
        result = self.extractor._extract_key_phrase(narrative)
        assert result != ""

    def test_no_match_returns_empty(self):
        narrative = "A simple uneventful scene with nobody around."
        result = self.extractor._extract_key_phrase(narrative)
        assert result == ""

    def test_returns_string(self):
        result = self.extractor._extract_key_phrase("anything")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _generate_concept
# ---------------------------------------------------------------------------


class TestGenerateConcept:
    def setup_method(self):
        self.extractor = MusicExtractor()

    def test_result_at_least_100_chars(self):
        room = _make_room()
        concept = self.extractor._generate_concept(room, "A quiet scene.")
        assert len(concept) >= 100

    def test_room_name_in_concept(self):
        room = _make_room(name="The Obsidian Spire")
        concept = self.extractor._generate_concept(room, "")
        assert "Obsidian Spire" in concept

    def test_inhabitants_mentioned(self):
        room = _make_room(inhabitants=["The Pale Watcher", "Echo Ghost"])
        concept = self.extractor._generate_concept(room, "")
        assert "Pale Watcher" in concept or "Echo Ghost" in concept

    def test_no_inhabitants_still_works(self):
        room = _make_room(inhabitants=[])
        concept = self.extractor._generate_concept(room, "Quiet.")
        assert isinstance(concept, str)
        assert len(concept) >= 100


# ---------------------------------------------------------------------------
# extract_song_proposal (integration)
# ---------------------------------------------------------------------------


class TestExtractSongProposal:
    def setup_method(self):
        self.extractor = MusicExtractor()
        random.seed(42)

    def test_returns_song_proposal_iteration(self):
        room = _make_room()
        result = self.extractor.extract_song_proposal(room, "A tense narrative.")
        assert isinstance(result, SongProposalIteration)

    def test_title_matches_room_name(self):
        room = _make_room(name="The Void Atrium")
        result = self.extractor.extract_song_proposal(room, "")
        assert result.title == "The Void Atrium"

    def test_rainbow_color_is_y(self):
        room = _make_room()
        result = self.extractor.extract_song_proposal(room, "")
        assert result.rainbow_color == "Y"

    def test_tempo_is_4_4(self):
        room = _make_room()
        result = self.extractor.extract_song_proposal(room, "")
        assert result.tempo == "4/4"

    def test_bpm_in_range(self):
        room = _make_room()
        result = self.extractor.extract_song_proposal(room, "")
        assert 40 <= result.bpm <= 200
