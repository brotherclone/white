"""Tests for retrieve_samples.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import yaml

from app.generators.midi.production.retrieve_samples import (
    retrieve_by_clap_similarity,
    retrieve_by_color,
    write_sample_map,
    copy_audio_files,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_df(n_red: int = 5, n_blue: int = 3, seed: int = 42) -> pd.DataFrame:
    """Build a minimal fake CLAP index DataFrame."""
    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n_red):
        rows.append(
            {
                "segment_id": f"red_seg_{i:03d}",
                "song_slug": f"Red Song {i}",
                "color": "Red",
                "clap_embedding": rng.random(512).astype(np.float32),
                "audio_path": f"/media/red_{i}.wav",
                "temporal": None,
                "spatial": None,
                "ontological": None,
                "confidence": None,
            }
        )

    for i in range(n_blue):
        rows.append(
            {
                "segment_id": f"blue_seg_{i:03d}",
                "song_slug": f"Blue Song {i}",
                "color": "Blue",
                "clap_embedding": rng.random(512).astype(np.float32),
                "audio_path": f"/media/blue_{i}.wav",
                "temporal": None,
                "spatial": None,
                "ontological": None,
                "confidence": None,
            }
        )

    return pd.DataFrame(rows)


def _make_mock_scorer(match_values: list[float] | None = None):
    """Build a Refractor mock returning controllable chromatic distributions."""
    scorer = MagicMock()

    call_count = [0]
    vals = match_values or []

    def fake_score(audio_emb=None, concept_emb=None):
        idx = call_count[0]
        call_count[0] += 1
        # Use provided match value to build a distribution that yields it
        # Temporal: Red target is [0.8, 0.1, 0.1]
        # Returning high "past" value drives high chromatic_match
        strength = vals[idx] if idx < len(vals) else 0.5
        return {
            "temporal": {
                "past": strength,
                "present": (1 - strength) / 2,
                "future": (1 - strength) / 2,
            },
            "spatial": {
                "thing": strength,
                "place": (1 - strength) / 2,
                "person": (1 - strength) / 2,
            },
            "ontological": {
                "imagined": (1 - strength) / 2,
                "forgotten": (1 - strength) / 2,
                "known": strength,
            },
            "confidence": 0.9,
        }

    scorer.score.side_effect = fake_score
    return scorer


# ---------------------------------------------------------------------------
# 5.1 Unit: retrieve_by_color — correct top-N sorted by match
# ---------------------------------------------------------------------------


class TestRetrieveByColor:

    def test_returns_top_n_results(self):
        df = _make_df(n_red=5, n_blue=3)
        scorer = _make_mock_scorer([0.9, 0.7, 0.5, 0.3, 0.1])
        results = retrieve_by_color(df, "Red", top_n=3, _scorer=scorer)
        assert len(results) == 3

    def test_sorted_descending_by_match(self):
        df = _make_df(n_red=5)
        scorer = _make_mock_scorer([0.9, 0.3, 0.7, 0.1, 0.5])
        results = retrieve_by_color(df, "Red", top_n=5, _scorer=scorer)
        matches = [r["match"] for r in results]
        assert matches == sorted(matches, reverse=True)

    def test_ranks_are_sequential_from_one(self):
        df = _make_df(n_red=4)
        scorer = _make_mock_scorer([0.8, 0.6, 0.4, 0.2])
        results = retrieve_by_color(df, "Red", top_n=4, _scorer=scorer)
        assert [r["rank"] for r in results] == [1, 2, 3, 4]

    def test_filters_to_requested_color_only(self):
        df = _make_df(n_red=3, n_blue=3)
        scorer = _make_mock_scorer([0.9, 0.8, 0.7])
        results = retrieve_by_color(df, "Red", top_n=10, _scorer=scorer)
        assert all(r["color"] == "Red" for r in results)
        assert len(results) == 3

    def test_result_fields_present(self):
        df = _make_df(n_red=2)
        scorer = _make_mock_scorer([0.8, 0.4])
        results = retrieve_by_color(df, "Red", top_n=2, _scorer=scorer)
        for r in results:
            assert "rank" in r
            assert "segment_id" in r
            assert "song_slug" in r
            assert "color" in r
            assert "match" in r
            assert "audio_path" in r

    def test_returns_all_if_fewer_than_top_n(self):
        df = _make_df(n_red=2, n_blue=3)
        scorer = _make_mock_scorer([0.8, 0.4])
        results = retrieve_by_color(df, "Red", top_n=10, _scorer=scorer)
        assert len(results) == 2

    def test_raises_on_invalid_color(self):
        df = _make_df(n_red=2)
        import pytest

        with pytest.raises(ValueError, match="Unknown color"):
            retrieve_by_color(df, "Chartreuse", top_n=5)

    def test_empty_result_for_color_with_no_segments(self):
        df = _make_df(n_red=3, n_blue=0)
        results = retrieve_by_color(df, "Blue", top_n=5)
        assert results == []


# ---------------------------------------------------------------------------
# 5.2 Unit: retrieve_by_clap_similarity — correct cosine ranking
# ---------------------------------------------------------------------------


class TestRetrieveByClapSimilarity:

    def _make_df_with_known_embeddings(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Build df where the first segment is most similar to a fixed query."""
        rng = np.random.default_rng(0)
        query = np.ones(512, dtype=np.float32)
        # First embedding: same direction as query → highest similarity
        # Others: random → lower similarity
        embs = [query.copy()]  # sim ≈ 1.0
        for _ in range(4):
            embs.append(rng.random(512).astype(np.float32))

        rows = [
            {
                "segment_id": f"seg_{i:03d}",
                "song_slug": f"Song {i}",
                "color": "Red",
                "clap_embedding": e,
                "audio_path": f"/media/{i}.wav",
            }
            for i, e in enumerate(embs)
        ]
        return pd.DataFrame(rows), query

    def test_top_result_is_most_similar(self):
        df, query = self._make_df_with_known_embeddings()
        results = retrieve_by_clap_similarity(df, query, top_n=5)
        assert results[0]["segment_id"] == "seg_000"

    def test_similarities_descending(self):
        df, query = self._make_df_with_known_embeddings()
        results = retrieve_by_clap_similarity(df, query, top_n=5)
        sims = [r["similarity"] for r in results]
        assert sims == sorted(sims, reverse=True)

    def test_ranks_sequential_from_one(self):
        df, query = self._make_df_with_known_embeddings()
        results = retrieve_by_clap_similarity(df, query, top_n=3)
        assert [r["rank"] for r in results] == [1, 2, 3]

    def test_result_fields_present(self):
        df, query = self._make_df_with_known_embeddings()
        results = retrieve_by_clap_similarity(df, query, top_n=2)
        for r in results:
            assert "rank" in r
            assert "segment_id" in r
            assert "color" in r
            assert "similarity" in r
            assert "audio_path" in r

    def test_respects_top_n(self):
        df, query = self._make_df_with_known_embeddings()
        results = retrieve_by_clap_similarity(df, query, top_n=2)
        assert len(results) == 2

    def test_cross_color_includes_all_colors(self):
        rng = np.random.default_rng(42)
        rows = [
            {
                "segment_id": f"s{i}",
                "song_slug": "",
                "color": c,
                "clap_embedding": rng.random(512).astype(np.float32),
                "audio_path": None,
            }
            for i, c in enumerate(["Red", "Blue", "Green", "Violet", "Orange"])
        ]
        df = pd.DataFrame(rows)
        query = rng.random(512).astype(np.float32)
        results = retrieve_by_clap_similarity(df, query, top_n=5)
        colors_seen = {r["color"] for r in results}
        assert len(colors_seen) > 1  # cross-color

    def test_similarity_for_identical_embedding_near_one(self):
        rng = np.random.default_rng(1)
        base = rng.random(512).astype(np.float32)
        rows = [
            {
                "segment_id": "exact",
                "song_slug": "",
                "color": "Red",
                "clap_embedding": base.copy(),
                "audio_path": None,
            },
            {
                "segment_id": "other",
                "song_slug": "",
                "color": "Blue",
                "clap_embedding": rng.random(512).astype(np.float32),
                "audio_path": None,
            },
        ]
        df = pd.DataFrame(rows)
        results = retrieve_by_clap_similarity(df, base, top_n=1)
        assert results[0]["segment_id"] == "exact"
        assert results[0]["similarity"] > 0.99


# ---------------------------------------------------------------------------
# 5.3 Unit: write_sample_map — valid YAML with expected fields
# ---------------------------------------------------------------------------


class TestWriteSampleMap:

    def _make_results(self) -> list[dict]:
        return [
            {
                "rank": 1,
                "segment_id": "seg_001",
                "song_slug": "Red Song",
                "color": "Red",
                "match": 0.87,
                "audio_path": "/media/seg_001.wav",
            },
            {
                "rank": 2,
                "segment_id": "seg_002",
                "song_slug": "Red Song",
                "color": "Red",
                "match": 0.72,
                "audio_path": None,
            },
        ]

    def test_file_is_written(self, tmp_path):
        out = write_sample_map(self._make_results(), tmp_path, "Red")
        assert out.exists()
        assert out.name == "sample_map.yml"

    def test_yaml_has_required_header_fields(self, tmp_path):
        write_sample_map(self._make_results(), tmp_path, "Red")
        loaded = yaml.safe_load((tmp_path / "sample_map.yml").read_text())
        assert loaded["color"] == "Red"
        assert "generated" in loaded
        assert loaded["count"] == 2

    def test_results_list_preserved(self, tmp_path):
        write_sample_map(self._make_results(), tmp_path, "Red")
        loaded = yaml.safe_load((tmp_path / "sample_map.yml").read_text())
        assert len(loaded["results"]) == 2
        assert loaded["results"][0]["segment_id"] == "seg_001"
        assert loaded["results"][0]["match"] == 0.87

    def test_null_audio_path_serialized_correctly(self, tmp_path):
        write_sample_map(self._make_results(), tmp_path, "Red")
        loaded = yaml.safe_load((tmp_path / "sample_map.yml").read_text())
        assert loaded["results"][1]["audio_path"] is None

    def test_output_dir_created_if_missing(self, tmp_path):
        nested = tmp_path / "deep" / "dir"
        write_sample_map(self._make_results(), nested, "Blue")
        assert (nested / "sample_map.yml").exists()

    def test_empty_results_written_cleanly(self, tmp_path):
        out = write_sample_map([], tmp_path, "Green")
        loaded = yaml.safe_load(out.read_text())
        assert loaded["count"] == 0
        assert loaded["results"] == []


# ---------------------------------------------------------------------------
# 5.4 Integration: stub parquet, verify CLI write with no media copy
# ---------------------------------------------------------------------------


class TestIntegrationRetrieve:

    def _make_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(99)
        rows = []
        for i in range(6):
            rows.append(
                {
                    "segment_id": f"red_seg_{i:03d}",
                    "song_slug": "Red Song",
                    "color": "Red",
                    "clap_embedding": rng.random(512).astype(np.float32),
                    "audio_path": None,
                    "temporal": None,
                    "spatial": None,
                    "ontological": None,
                    "confidence": None,
                }
            )
        return pd.DataFrame(rows)

    def test_end_to_end_writes_sample_map(self, tmp_path):
        df = self._make_df()
        scorer = _make_mock_scorer([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])

        results = retrieve_by_color(df, "Red", top_n=3, _scorer=scorer)
        out = write_sample_map(results, tmp_path, "Red")

        assert out.exists()
        loaded = yaml.safe_load(out.read_text())
        assert loaded["color"] == "Red"
        assert loaded["count"] == 3
        assert len(loaded["results"]) == 3

    def test_results_ranked_correctly(self, tmp_path):
        df = self._make_df()
        scorer = _make_mock_scorer([0.4, 0.9, 0.2, 0.7, 0.6, 0.1])

        results = retrieve_by_color(df, "Red", top_n=6, _scorer=scorer)

        # Verify descending match order
        matches = [r["match"] for r in results]
        assert matches == sorted(matches, reverse=True)

    def test_copy_audio_skips_missing_files(self, tmp_path):
        results = [
            {"rank": 1, "segment_id": "s1", "audio_path": "/nonexistent/s1.wav"},
            {"rank": 2, "segment_id": "s2", "audio_path": None},
        ]
        n = copy_audio_files(results, tmp_path)
        assert n == 0
        # audio dir still created
        assert (tmp_path / "audio").exists()

    def test_copy_audio_cuts_from_source_when_segment_missing(self, tmp_path):
        """Falls back to cutting from source WAV when segment WAV doesn't exist."""
        import soundfile as sf
        import numpy as np

        # Write a real WAV source (1 second of silence at 16kHz)
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        src = src_dir / "source_track.wav"
        sf.write(str(src), np.zeros(16000, dtype=np.float32), 16000)

        results = [
            {
                "rank": 1,
                "segment_id": "seg_001",
                "audio_path": "/nonexistent/seg_001.wav",  # won't exist
                "source_audio_file": str(src),
                "start_seconds": 0.1,
                "end_seconds": 0.5,
            }
        ]
        out_dir = tmp_path / "output"
        n = copy_audio_files(results, out_dir)

        assert n == 1
        out_wav = out_dir / "audio" / "seg_001.wav"
        assert out_wav.exists()
        data, sr = sf.read(str(out_wav))
        assert sr == 16000
        assert abs(len(data) / sr - 0.4) < 0.01  # ~0.4s

    def test_copy_audio_copies_existing_files(self, tmp_path):
        # Create real audio files
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        f1 = src_dir / "segment_1.wav"
        f2 = src_dir / "segment_2.wav"
        f1.write_bytes(b"RIFF")
        f2.write_bytes(b"RIFF")

        results = [
            {"rank": 1, "segment_id": "s1", "audio_path": str(f1)},
            {"rank": 2, "segment_id": "s2", "audio_path": str(f2)},
        ]
        out_dir = tmp_path / "output"
        n = copy_audio_files(results, out_dir)

        assert n == 2
        # Output filenames use segment_id, not source filename
        assert (out_dir / "audio" / "s1.wav").exists()
        assert (out_dir / "audio" / "s2.wav").exists()
