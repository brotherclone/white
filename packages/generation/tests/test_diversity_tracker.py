"""Tests for the album-level template diversity tracker."""

from __future__ import annotations

import pytest

from white_generation.util.diversity_tracker import (
    diversity_factor,
    find_album_dir,
    load_registry,
    record_use,
    save_registry,
)


class TestDiversityFactor:
    def test_zero_uses_gets_bonus(self):
        assert diversity_factor("motorik", {}) == 1.15

    def test_one_use_is_neutral(self):
        assert diversity_factor("motorik", {"motorik": 1}) == 1.0

    def test_two_uses_penalised_at_0_6(self):
        assert diversity_factor("motorik", {"motorik": 2}) == 0.6

    def test_penalty_deepens_with_further_reuse(self):
        assert diversity_factor("motorik", {"motorik": 3}) == pytest.approx(0.5)
        assert diversity_factor("motorik", {"motorik": 4}) == pytest.approx(0.4)

    def test_penalty_floors_at_0_35(self):
        assert diversity_factor("motorik", {"motorik": 5}) == 0.35
        assert diversity_factor("motorik", {"motorik": 10}) == 0.35

    def test_unknown_template_treated_as_zero_uses(self):
        assert diversity_factor("never_used", {"motorik": 3}) == 1.15


class TestRegistryMissing:
    def test_missing_registry_file_yields_empty_dict(self, tmp_path):
        registry = load_registry(tmp_path)
        assert registry == {}

    def test_missing_registry_means_no_penalty(self, tmp_path):
        registry = load_registry(tmp_path)
        assert diversity_factor("anything", registry) == 1.15


class TestLoadSaveRegistry:
    def test_round_trip(self, tmp_path):
        save_registry(tmp_path, {"motorik": 2})
        assert load_registry(tmp_path) == {"motorik": 2}

    def test_corrupt_file_yields_empty_dict(self, tmp_path):
        (tmp_path / "used_templates.json").write_text("not json")
        assert load_registry(tmp_path) == {}


class TestRecordUse:
    def test_increments_existing_count(self):
        registry = {"motorik": 1}
        record_use("motorik", registry)
        assert registry["motorik"] == 2

    def test_adds_new_template_at_one(self):
        registry: dict[str, int] = {}
        record_use("motorik", registry)
        assert registry["motorik"] == 1


class TestFindAlbumDir:
    def test_finds_shrink_wrapped_ancestor(self, tmp_path):
        album_dir = tmp_path / "shrink_wrapped"
        nested = album_dir / "thread" / "production" / "song"
        nested.mkdir(parents=True)
        assert find_album_dir(nested) == album_dir

    def test_returns_none_when_not_found(self, tmp_path):
        assert find_album_dir(tmp_path) is None
