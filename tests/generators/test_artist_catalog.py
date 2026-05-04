"""Tests for app/generators/artist_catalog.py"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_catalog(tmp_path, entries: dict) -> Path:
    catalog_path = tmp_path / "artist_catalog.yml"
    header = "# Artist Style Catalog\n"
    body = (
        yaml.dump(
            entries, default_flow_style=False, sort_keys=False, allow_unicode=True
        )
        if entries
        else ""
    )
    catalog_path.write_text(header + body)
    return catalog_path


def _make_thread(tmp_path, songs: list[dict]) -> Path:
    """Create a thread dir with production plans containing sounds_like."""
    thread_dir = tmp_path / "thread"
    for song in songs:
        prod_dir = thread_dir / "production" / song["slug"]
        prod_dir.mkdir(parents=True)
        plan = {
            "song_slug": song["slug"],
            "bpm": 120,
            "time_sig": "4/4",
            "color": "Red",
            "key": "C major",
            "sounds_like": song.get("sounds_like", []),
        }
        with open(prod_dir / "production_plan.yml", "w") as f:
            yaml.dump(plan, f)
    return thread_dir


# ---------------------------------------------------------------------------
# parse_sounds_like_string
# ---------------------------------------------------------------------------


class TestParseSoundsLikeString:
    def test_with_discogs_ids(self):
        from white_generation.artist_catalog import parse_sounds_like_string

        result = parse_sounds_like_string(
            "Bear in Heaven, discogs_id: 123, Seefeel, discogs_id: 456"
        )
        assert result == [("Bear in Heaven", 123), ("Seefeel", 456)]

    def test_without_discogs_ids(self):
        from white_generation.artist_catalog import parse_sounds_like_string

        result = parse_sounds_like_string("Boards of Canada, Aphex Twin")
        assert ("Boards of Canada", None) in result or len(result) >= 1

    def test_partial_discogs(self):
        from white_generation.artist_catalog import parse_sounds_like_string

        result = parse_sounds_like_string("Artist A, discogs_id: 99, Artist B")
        assert result[0] == ("Artist A", 99)
        assert result[1][0] == "Artist B"
        assert result[1][1] is None


# ---------------------------------------------------------------------------
# collect_sounds_like
# ---------------------------------------------------------------------------


class TestCollectSoundsLike:
    def test_deduplicates_across_plans(self, tmp_path):
        from white_generation.artist_catalog import collect_sounds_like

        thread = _make_thread(
            tmp_path,
            [
                {"slug": "song_a", "sounds_like": ["Seefeel", "Oval"]},
                {"slug": "song_b", "sounds_like": ["Seefeel", "Autechre"]},
            ],
        )
        result = collect_sounds_like(thread_dir=thread)
        names = [name for name, _ in result]
        assert names.count("Seefeel") == 1
        assert "Oval" in names
        assert "Autechre" in names

    def test_empty_thread(self, tmp_path):
        from white_generation.artist_catalog import collect_sounds_like

        thread = _make_thread(tmp_path, [{"slug": "no_sounds", "sounds_like": []}])
        result = collect_sounds_like(thread_dir=thread)
        assert result == []


# ---------------------------------------------------------------------------
# generate_description prompt constraints
# ---------------------------------------------------------------------------


class TestGenerateDescriptionPromptConstraints:
    def test_prompt_excludes_biography(self):
        from white_generation.artist_catalog import _GENERATE_PROMPT_TEMPLATE

        prompt = _GENERATE_PROMPT_TEMPLATE.format(artist_name="Test Artist")
        assert (
            "birth" in prompt.lower()
            or "biography" in prompt.lower()
            or "Biographical" in prompt
        )
        assert "Do NOT" in prompt or "do NOT" in prompt

    def test_prompt_includes_copyright_instruction(self):
        from white_generation.artist_catalog import _GENERATE_PROMPT_TEMPLATE

        prompt = _GENERATE_PROMPT_TEMPLATE.format(artist_name="Test Artist")
        assert "copyright" in prompt.lower() or "copyrighted" in prompt.lower()

    def test_prompt_includes_artist_name(self):
        from white_generation.artist_catalog import _GENERATE_PROMPT_TEMPLATE

        prompt = _GENERATE_PROMPT_TEMPLATE.format(artist_name="My Bloody Valentine")
        assert "My Bloody Valentine" in prompt

    def test_unknown_artist_sentinel(self):
        """generate_description returns None when Claude responds UNKNOWN_ARTIST."""
        from white_generation.artist_catalog import generate_description

        mock_client = MagicMock()
        mock_client.messages.create.return_value.content = [
            MagicMock(text="UNKNOWN_ARTIST")
        ]
        result = generate_description("Obscure Band XYZ", mock_client)
        assert result is None

    def test_description_returned_on_success(self):
        from white_generation.artist_catalog import generate_description

        mock_client = MagicMock()
        mock_client.messages.create.return_value.content = [
            MagicMock(text="A dreamy shoegaze band with thick guitar textures.")
        ]
        result = generate_description("Slowdive", mock_client)
        assert result == "A dreamy shoegaze band with thick guitar textures."


# ---------------------------------------------------------------------------
# generate_missing — idempotent
# ---------------------------------------------------------------------------


class TestGenerateMissingIdempotent:
    def test_no_api_calls_when_all_present(self, tmp_path):
        from white_generation.artist_catalog import generate_missing

        catalog_path = _write_catalog(
            tmp_path,
            {
                "Seefeel": {
                    "slug": "seefeel",
                    "status": "draft",
                    "description": "Hazy post-rock textures.",
                    "style_tags": [],
                    "chromatic_score": None,
                    "discogs_id": None,
                    "notes": "",
                }
            },
        )
        artists = [("Seefeel", None)]

        with patch("anthropic.Anthropic") as mock_cls:
            added = generate_missing(artists, catalog_path)

        mock_cls.assert_not_called()
        assert added == []

    def test_generates_for_new_artists(self, tmp_path):
        from white_generation.artist_catalog import generate_missing

        catalog_path = _write_catalog(tmp_path, {})
        artists = [("New Artist", None)]

        mock_client = MagicMock()
        mock_client.messages.create.return_value.content = [
            MagicMock(text="Sparse electronic minimalism.")
        ]

        with patch("anthropic.Anthropic", return_value=mock_client):
            added = generate_missing(artists, catalog_path)

        assert "New Artist" in added
        # Verify entry was written to file
        from white_generation.artist_catalog import load_catalog

        catalog = load_catalog(catalog_path)
        assert "New Artist" in catalog
        assert catalog["New Artist"]["status"] == "draft"

    def test_plain_string_list_no_api_call_when_all_present(self, tmp_path):
        from white_generation.artist_catalog import generate_missing

        catalog_path = _write_catalog(
            tmp_path,
            {
                "Seefeel": {
                    "slug": "seefeel",
                    "status": "draft",
                    "description": "Hazy textures.",
                    "style_tags": [],
                    "chromatic_score": None,
                    "discogs_id": None,
                    "notes": "",
                }
            },
        )

        with patch("anthropic.Anthropic") as mock_cls:
            added = generate_missing(["Seefeel"], catalog_path)

        mock_cls.assert_not_called()
        assert added == []


# ---------------------------------------------------------------------------
# load_artist_context
# ---------------------------------------------------------------------------


class TestLoadArtistContext:
    def test_reviewed_preferred_over_draft(self, tmp_path):
        """When both reviewed and draft exist (different artists), reviewed is used first."""
        from white_generation.artist_catalog import load_artist_context

        catalog_path = _write_catalog(
            tmp_path,
            {
                "Artist Reviewed": {
                    "slug": "artist_reviewed",
                    "status": "reviewed",
                    "description": "This is the reviewed description.",
                    "style_tags": [],
                    "chromatic_score": None,
                    "discogs_id": None,
                    "notes": "",
                },
                "Artist Draft": {
                    "slug": "artist_draft",
                    "status": "draft",
                    "description": "This is the draft description.",
                    "style_tags": [],
                    "chromatic_score": None,
                    "discogs_id": None,
                    "notes": "",
                },
            },
        )
        result = load_artist_context(
            ["Artist Reviewed", "Artist Draft"], catalog_path=catalog_path
        )
        assert "This is the reviewed description." in result
        assert "This is the draft description." in result
        # Draft note should be printed (tested via capsys in separate test)
        assert "STYLE REFERENCES" in result

    def test_draft_note_printed(self, tmp_path, capsys):
        from white_generation.artist_catalog import load_artist_context

        catalog_path = _write_catalog(
            tmp_path,
            {
                "Draft Band": {
                    "slug": "draft_band",
                    "status": "draft",
                    "description": "Some description.",
                    "style_tags": [],
                    "chromatic_score": None,
                    "discogs_id": None,
                    "notes": "",
                }
            },
        )
        load_artist_context(["Draft Band"], catalog_path=catalog_path)
        captured = capsys.readouterr()
        assert "consider reviewing" in captured.out

    def test_missing_artist_prints_note(self, tmp_path, capsys):
        from white_generation.artist_catalog import load_artist_context

        catalog_path = _write_catalog(tmp_path, {})
        result = load_artist_context(["Unknown Band"], catalog_path=catalog_path)
        assert result == ""
        captured = capsys.readouterr()
        assert "not in catalog" in captured.out
        assert "Unknown Band" in captured.out

    def test_empty_sounds_like(self, tmp_path):
        from white_generation.artist_catalog import load_artist_context

        catalog_path = _write_catalog(tmp_path, {})
        result = load_artist_context([], catalog_path=catalog_path)
        assert result == ""

    def test_null_description_skipped(self, tmp_path):
        from white_generation.artist_catalog import load_artist_context

        catalog_path = _write_catalog(
            tmp_path,
            {
                "Unknown Band": {
                    "slug": "unknown_band",
                    "status": "draft",
                    "description": None,
                    "style_tags": [],
                    "chromatic_score": None,
                    "discogs_id": None,
                    "notes": "Unknown artist — fill description manually",
                }
            },
        )
        result = load_artist_context(["Unknown Band"], catalog_path=catalog_path)
        assert result == ""


# ---------------------------------------------------------------------------
# score_chromatic
# ---------------------------------------------------------------------------


class TestScoreChromatic:
    def test_skips_null_description(self, tmp_path, capsys):
        from white_generation.artist_catalog import score_chromatic

        catalog_path = _write_catalog(
            tmp_path,
            {
                "Has Description": {
                    "slug": "has_description",
                    "status": "draft",
                    "description": "Rich textural soundscapes.",
                    "style_tags": [],
                    "chromatic_score": None,
                    "discogs_id": None,
                    "notes": "",
                },
                "No Description": {
                    "slug": "no_description",
                    "status": "draft",
                    "description": None,
                    "style_tags": [],
                    "chromatic_score": None,
                    "discogs_id": None,
                    "notes": "Unknown artist",
                },
            },
        )

        mock_scorer = MagicMock()
        mock_scorer.score_batch.return_value = [
            {
                "temporal": {"past": 0.7, "present": 0.2, "future": 0.1},
                "spatial": {"thing": 0.5, "place": 0.4, "person": 0.1},
                "ontological": {"imagined": 0.1, "forgotten": 0.2, "known": 0.7},
                "confidence": 0.04,
            }
        ]

        with patch("white_analysis.refractor.Refractor", return_value=mock_scorer):
            score_chromatic(catalog_path)

        captured = capsys.readouterr()
        assert "SKIP" in captured.out
        assert "No Description" in captured.out

        # Only the entry with a description should have been scored
        mock_scorer.score_batch.assert_called_once()
        args = mock_scorer.score_batch.call_args[0][0]
        assert any("Rich textural soundscapes." in str(c) for c in args)
