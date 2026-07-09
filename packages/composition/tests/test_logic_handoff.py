"""Tests for logic_handoff.handoff() -- arrangement.txt placeholder + sync behavior."""

import os
from pathlib import Path
from unittest.mock import patch

import yaml


def _write_song_context(prod_dir: Path, title: str = "Test Song") -> None:
    prod_dir.mkdir(parents=True, exist_ok=True)
    with open(prod_dir / "song_context.yml", "w") as f:
        yaml.dump(
            {"title": title, "thread": ""},
            f,
            sort_keys=False,
            allow_unicode=True,
            width=float("inf"),
        )


class TestHandoffArrangementPlaceholder:
    def test_creates_blank_arrangement_when_absent(self, tmp_path):
        from white_composition.logic_handoff import handoff

        prod_dir = tmp_path / "production" / "song_a"
        _write_song_context(prod_dir)
        logic_dir = tmp_path / "logic_output"

        with patch.dict(os.environ, {"LOGIC_OUTPUT_DIR": str(logic_dir)}):
            song_dir = handoff(prod_dir)

        arrangement = song_dir / "arrangement.txt"
        assert arrangement.exists()
        assert arrangement.stat().st_size == 0

    def test_does_not_overwrite_existing_logic_arrangement(self, tmp_path):
        from white_composition.logic_handoff import handoff, resolve_song_dir

        prod_dir = tmp_path / "production" / "song_a"
        _write_song_context(prod_dir)
        logic_dir = tmp_path / "logic_output"

        with patch.dict(os.environ, {"LOGIC_OUTPUT_DIR": str(logic_dir)}):
            song_dir = resolve_song_dir(prod_dir)
            song_dir.mkdir(parents=True)
            (song_dir / "arrangement.txt").write_text("real arrangement content\n")

            handoff(prod_dir)

        assert (
            song_dir / "arrangement.txt"
        ).read_text() == "real arrangement content\n"

    def test_blank_placeholder_not_synced_to_production_dir(self, tmp_path):
        """A freshly-created blank Logic arrangement.txt must not overwrite (or
        create) production_dir/arrangement.txt."""
        from white_composition.logic_handoff import handoff

        prod_dir = tmp_path / "production" / "song_a"
        _write_song_context(prod_dir)
        logic_dir = tmp_path / "logic_output"

        with patch.dict(os.environ, {"LOGIC_OUTPUT_DIR": str(logic_dir)}):
            handoff(prod_dir)

        assert not (prod_dir / "arrangement.txt").exists()

    def test_real_logic_arrangement_still_syncs_back(self, tmp_path):
        """Regression: a real (non-empty) Logic arrangement.txt must still sync
        back into the production directory, as before."""
        from white_composition.logic_handoff import handoff, resolve_song_dir

        prod_dir = tmp_path / "production" / "song_a"
        _write_song_context(prod_dir)
        logic_dir = tmp_path / "logic_output"

        with patch.dict(os.environ, {"LOGIC_OUTPUT_DIR": str(logic_dir)}):
            song_dir = resolve_song_dir(prod_dir)
            song_dir.mkdir(parents=True)
            (song_dir / "arrangement.txt").write_text(
                "01:00:00:00.00\tverse\t4\t00:00:08:00.00\n"
            )

            handoff(prod_dir)

        assert (prod_dir / "arrangement.txt").exists()
        assert "verse" in (prod_dir / "arrangement.txt").read_text()
