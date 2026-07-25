"""Tests for playlist_sync."""

import numpy as np
import soundfile as sf
import yaml

from white_composition.lp_sides import SidesDocument, assign_song
from white_composition.playlist_sync import (
    DEFAULT_OUTPUT_DIR,
    PlaylistConfig,
    classify_songs,
    load_playlist_config,
    material_produced_seconds,
    save_playlist_config,
    sync_playlists,
)


def _write_mp3(path, seconds: float = 2.0, samplerate: int = 8000):
    samples = np.zeros(int(seconds * samplerate), dtype="float32")
    sf.write(str(path), samples, samplerate)


def _make_song(
    tmp_path,
    song_id: str,
    title: str,
    *,
    has_mix: bool = True,
    lifecycle_status: str | None = None,
    lp_consideration: str = "not_considered",
    mix_ext: str = ".mp3",
    production_slug: str | None = None,
) -> dict:
    prod_dir = tmp_path / "productions" / song_id
    prod_dir.mkdir(parents=True, exist_ok=True)
    ctx: dict = {}
    if has_mix:
        mix_path = prod_dir / f"mix{mix_ext}"
        mix_path.write_bytes(b"fake audio bytes")
        ctx["mix_file"] = str(mix_path)
    with open(prod_dir / "song_context.yml", "w") as f:
        yaml.dump(ctx, f)
    return {
        "id": song_id,
        "production_slug": production_slug or song_id,
        "production_path": str(prod_dir),
        "title": title,
        "has_mix": has_mix,
        "lifecycle_status": lifecycle_status,
        "lp_consideration": lp_consideration,
    }


class TestPlaylistConfig:
    def test_default_materialized_on_first_read(self, tmp_path):
        config = load_playlist_config(tmp_path)
        assert config.output_dir == DEFAULT_OUTPUT_DIR
        assert (tmp_path / "playlist_config.yml").exists()

    def test_round_trip(self, tmp_path):
        save_playlist_config(tmp_path, PlaylistConfig(output_dir="/tmp/custom"))
        loaded = load_playlist_config(tmp_path)
        assert loaded.output_dir == "/tmp/custom"


class TestClassifySongs:
    def test_rejects_bucket(self, tmp_path):
        scrapped = _make_song(
            tmp_path, "s1", "Scrapped Song", lifecycle_status="scrapped"
        )
        abandoned = _make_song(
            tmp_path, "s2", "Abandoned Song", lifecycle_status="abandoned"
        )
        result = classify_songs([scrapped, abandoned], SidesDocument.empty())
        ids = {s["id"] for s in result["rejects"]}
        assert ids == {"s1", "s2"}
        assert result["review"] == []
        assert result["wip"] == []

    def test_review_bucket(self, tmp_path):
        active = _make_song(tmp_path, "s1", "Active Song")
        merged = _make_song(tmp_path, "s2", "Merged Song", lifecycle_status="merged")
        result = classify_songs([active, merged], SidesDocument.empty())
        ids = {s["id"] for s in result["review"]}
        assert ids == {"s1", "s2"}

    def test_placed_excluded_from_review_even_if_rejected_status(self, tmp_path):
        """lp_consideration == placed always wins, regardless of lifecycle_status."""
        song = _make_song(
            tmp_path,
            "s1",
            "Placed Song",
            lifecycle_status="scrapped",
            lp_consideration="placed",
        )
        result = classify_songs([song], SidesDocument.empty())
        assert result["rejects"] == []
        assert result["review"] == []
        assert [s["id"] for s in result["wip"]] == ["s1"]

    def test_no_mix_excluded_from_all_buckets(self, tmp_path):
        song = _make_song(tmp_path, "s1", "No Mix", has_mix=False)
        result = classify_songs([song], SidesDocument.empty())
        assert result["rejects"] == []
        assert result["review"] == []
        assert result["wip"] == []

    def test_wip_ordered_by_side_and_position(self, tmp_path):
        song_a = _make_song(tmp_path, "a", "Song A", lp_consideration="placed")
        song_b = _make_song(tmp_path, "b", "Song B", lp_consideration="placed")
        song_c = _make_song(tmp_path, "c", "Song C", lp_consideration="placed")
        doc = SidesDocument.empty()
        assign_song(doc, "b", "A", 0, 100.0)
        assign_song(doc, "c", "A", 1, 100.0)
        assign_song(doc, "a", "B", 0, 100.0)

        result = classify_songs([song_a, song_b, song_c], doc)
        ordered_ids = [s["id"] for s in result["wip"]]
        assert ordered_ids == ["b", "c", "a"]
        assert result["wip"][0]["_side"] == "A"
        assert result["wip"][2]["_side"] == "B"

    def test_unsequenced_placed_song_appended_at_end(self, tmp_path):
        sequenced = _make_song(tmp_path, "seq", "Sequenced", lp_consideration="placed")
        unsequenced = _make_song(
            tmp_path, "unseq", "Unsequenced", lp_consideration="placed"
        )
        doc = SidesDocument.empty()
        assign_song(doc, "seq", "A", 0, 100.0)

        result = classify_songs([sequenced, unsequenced], doc)
        ordered_ids = [s["id"] for s in result["wip"]]
        assert ordered_ids == ["seq", "unseq"]
        assert result["wip"][1]["_side"] is None


class TestSyncPlaylists:
    def test_full_rebuild_populates_each_bucket(self, tmp_path):
        rejected = _make_song(tmp_path, "r1", "Rejected", lifecycle_status="scrapped")
        reviewed = _make_song(tmp_path, "v1", "Reviewed")
        placed = _make_song(tmp_path, "p1", "Placed", lp_consideration="placed")
        output_dir = tmp_path / "Listening"

        counts = sync_playlists(
            [rejected, reviewed, placed], SidesDocument.empty(), str(output_dir)
        )

        assert counts == {"rejects": 1, "review": 1, "wip": 1}
        assert (output_dir / "Rejects" / "REJECT_Rejected.mp3").exists()
        assert (output_dir / "Review" / "Reviewed.mp3").exists()
        assert (output_dir / "White Album WiP" / "99_unsequenced_Placed.mp3").exists()

    def test_rejects_filenames_are_prefixed_review_is_not(self, tmp_path):
        """Apple Music flattens folder structure on import, so Rejects and Review
        tracks with similar titles are otherwise indistinguishable once synced."""
        rejected = _make_song(tmp_path, "r1", "Overlap", lifecycle_status="scrapped")
        reviewed = _make_song(tmp_path, "v1", "Overlap Too")
        output_dir = tmp_path / "Listening"

        sync_playlists([rejected, reviewed], SidesDocument.empty(), str(output_dir))

        assert (output_dir / "Rejects" / "REJECT_Overlap.mp3").exists()
        assert (output_dir / "Review" / "Overlap Too.mp3").exists()

    def test_stale_file_removed_on_rebuild(self, tmp_path):
        output_dir = tmp_path / "Listening"
        song = _make_song(tmp_path, "r1", "Rejected", lifecycle_status="scrapped")

        sync_playlists([song], SidesDocument.empty(), str(output_dir))
        assert (output_dir / "Rejects" / "REJECT_Rejected.mp3").exists()

        # Song no longer matches any bucket (mix removed) -> file should disappear.
        song["has_mix"] = False
        sync_playlists([song], SidesDocument.empty(), str(output_dir))
        assert not (output_dir / "Rejects" / "REJECT_Rejected.mp3").exists()


class TestMaterialProducedSeconds:
    def test_sums_only_top_level_mp3s(self, tmp_path):
        output_dir = tmp_path / "Listening"
        output_dir.mkdir()
        _write_mp3(output_dir / "a.mp3", seconds=3.0)
        _write_mp3(output_dir / "b.mp3", seconds=2.0)
        (output_dir / "notes.txt").write_text("not audio")
        subdir = output_dir / "Rejects"
        subdir.mkdir()
        _write_mp3(subdir / "c.mp3", seconds=10.0)

        total, count = material_produced_seconds(str(output_dir))

        assert total == 5.0
        assert count == 2

    def test_missing_dir_returns_zero(self, tmp_path):
        total, count = material_produced_seconds(str(tmp_path / "nope"))
        assert (total, count) == (0.0, 0)

    def test_ignores_non_mp3_extensions(self, tmp_path):
        output_dir = tmp_path / "Listening"
        output_dir.mkdir()
        _write_mp3(output_dir / "a.wav", seconds=3.0)

        total, count = material_produced_seconds(str(output_dir))

        assert (total, count) == (0.0, 0)

    def test_unrelated_file_outside_subfolders_untouched(self, tmp_path):
        output_dir = tmp_path / "Listening"
        output_dir.mkdir(parents=True)
        sentinel = output_dir / "not_ours.txt"
        sentinel.write_text("leave me alone")

        song = _make_song(tmp_path, "r1", "Rejected", lifecycle_status="scrapped")
        sync_playlists([song], SidesDocument.empty(), str(output_dir))

        assert sentinel.exists()

    def test_non_audio_file_inside_subfolder_untouched(self, tmp_path):
        output_dir = tmp_path / "Listening"
        (output_dir / "Rejects").mkdir(parents=True)
        ds_store = output_dir / "Rejects" / ".DS_Store"
        ds_store.write_text("finder metadata")

        sync_playlists([], SidesDocument.empty(), str(output_dir))
        assert ds_store.exists()

    def test_title_collision_gets_suffixed(self, tmp_path):
        song_1 = _make_song(tmp_path, "dup1", "Same Title", lifecycle_status="scrapped")
        song_2 = _make_song(
            tmp_path, "dup2", "Same Title", lifecycle_status="abandoned"
        )
        output_dir = tmp_path / "Listening"
        sync_playlists([song_1, song_2], SidesDocument.empty(), str(output_dir))

        rejects_dir = output_dir / "Rejects"
        names = {p.name for p in rejects_dir.iterdir()}
        assert names == {"REJECT_Same Title_dup1.mp3", "REJECT_Same Title_dup2.mp3"}

    def test_wip_numeric_prefix_order(self, tmp_path):
        song_a = _make_song(tmp_path, "a", "Alpha", lp_consideration="placed")
        song_b = _make_song(tmp_path, "b", "Beta", lp_consideration="placed")
        doc = SidesDocument.empty()
        assign_song(doc, "b", "A", 0, 100.0)
        assign_song(doc, "a", "B", 0, 100.0)
        output_dir = tmp_path / "Listening"

        sync_playlists([song_a, song_b], doc, str(output_dir))

        wip_dir = output_dir / "White Album WiP"
        names = sorted(p.name for p in wip_dir.iterdir())
        assert names == ["01_A_Beta.mp3", "02_B_Alpha.mp3"]

    def test_unsequenced_wip_title_collision_gets_suffixed(self, tmp_path):
        """Two unsequenced placed songs sharing a title both get the fixed
        '99_unsequenced_' prefix — without collision suffixing they'd map to
        the same filename and one copy would silently overwrite the other."""
        song_1 = _make_song(tmp_path, "dup1", "Same Title", lp_consideration="placed")
        song_2 = _make_song(tmp_path, "dup2", "Same Title", lp_consideration="placed")
        output_dir = tmp_path / "Listening"

        counts = sync_playlists(
            [song_1, song_2], SidesDocument.empty(), str(output_dir)
        )

        assert counts["wip"] == 2
        wip_dir = output_dir / "White Album WiP"
        names = {p.name for p in wip_dir.iterdir()}
        assert names == {
            "99_unsequenced_Same Title_dup1.mp3",
            "99_unsequenced_Same Title_dup2.mp3",
        }

    def test_unchanged_file_not_recopied(self, tmp_path):
        song = _make_song(tmp_path, "r1", "Rejected", lifecycle_status="scrapped")
        output_dir = tmp_path / "Listening"
        sync_playlists([song], SidesDocument.empty(), str(output_dir))

        dest = output_dir / "Rejects" / "REJECT_Rejected.mp3"
        first_mtime = dest.stat().st_mtime_ns

        sync_playlists([song], SidesDocument.empty(), str(output_dir))
        assert dest.stat().st_mtime_ns == first_mtime

    def test_empty_output_dir_raises(self):
        try:
            sync_playlists([], SidesDocument.empty(), "")
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

    def test_home_directory_output_dir_raises(self):
        try:
            sync_playlists([], SidesDocument.empty(), "~")
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
