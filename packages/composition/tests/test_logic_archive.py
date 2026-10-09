"""Tests for logic_archive — classification, verified move, path rewrite, restore,
and archive-aware resolution in logic_handoff."""

import os
import unicodedata
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from white_composition import logic_archive
from white_composition.logic_archive import (
    ArchiveVerificationError,
    LogicArchiveOfflineError,
    archive_song,
    classify,
    load_manifest,
    main,
    restore_song,
    verify_copy,
)
from white_composition.logic_handoff import handoff, resolve_song_dir
from white_composition.lp_sides import SidesDocument, SideSong, save_sides
from white_core.enums.archive_classification import ArchiveClassification
from white_core.enums.archive_status import ArchiveStatus

THREAD = "white-thread-one"


def _dump(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, sort_keys=False, width=float("inf"))


def _load(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def env(tmp_path):
    album = tmp_path / "shrink_wrapped"
    logic_root = tmp_path / "Tracks"
    archive_root = tmp_path / "Archive" / "Extras"
    album.mkdir()
    logic_root.mkdir()
    archive_root.mkdir(parents=True)
    with patch.dict(
        os.environ,
        {"LOGIC_OUTPUT_DIR": str(logic_root), "LOGIC_ARCHIVE_DIR": str(archive_root)},
    ):
        yield album, logic_root, archive_root


def _make_song(album, logic_root, slug, title=None, thread=THREAD, extra_ctx=None):
    """Create a production dir and a populated Logic song folder."""
    title = title or slug.replace("_", " ").title()
    prod = album / thread / "production" / slug
    ctx = {"title": title, "thread": thread, **(extra_ctx or {})}
    _dump(ctx, prod / "song_context.yml")
    song_dir = logic_root / thread / f"{title} ({slug})"
    (song_dir / f"{title}.logicx" / "Alternatives").mkdir(parents=True)
    (song_dir / f"{title}.logicx" / "Alternatives" / "ProjectData").write_bytes(
        b"\x00logic" * 100
    )
    (song_dir / "MIDI" / "chords").mkdir(parents=True)
    (song_dir / "MIDI" / "chords" / "a.mid").write_bytes(b"MThd")
    _dump(
        {"song_title": title, "logic_project_path": str(song_dir)},
        song_dir / "composition.yml",
    )
    return prod, song_dir


def _place(album, *song_ids):
    doc = SidesDocument.empty()
    doc.sides["A"].songs = [SideSong(song_id=s, duration_seconds=1.0) for s in song_ids]
    save_sides(album, doc)


def _by_slug(folders, slug):
    return next(f for f in folders if f.folder_name.endswith(f"({slug})"))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestClassify:
    def test_placed_kept_candidate_unmatched(self, env):
        album, logic_root, _ = env
        _make_song(album, logic_root, "placed_v1")
        _make_song(album, logic_root, "kept_v1")
        _make_song(album, logic_root, "cand_v1")
        (logic_root / THREAD / "Orphan (gone_v1)").mkdir()
        (logic_root / THREAD / "No Suffix Here").mkdir()
        _place(album, f"{THREAD}__placed_v1")
        _dump({"songs": [f"{THREAD}__kept_v1"]}, album / "archive_keep.yml")

        folders = classify(logic_root, album)
        assert (
            _by_slug(folders, "placed_v1").classification
            == ArchiveClassification.PLACED
        )
        assert _by_slug(folders, "kept_v1").classification == ArchiveClassification.KEPT
        assert (
            _by_slug(folders, "cand_v1").classification
            == ArchiveClassification.CANDIDATE
        )
        assert (
            _by_slug(folders, "gone_v1").classification
            == ArchiveClassification.UNMATCHED
        )
        no_suffix = next(f for f in folders if f.folder_name == "No Suffix Here")
        assert no_suffix.classification == ArchiveClassification.UNMATCHED

    def test_keep_list_thread_protects_all_songs(self, env):
        album, logic_root, _ = env
        _make_song(album, logic_root, "a_v1")
        _make_song(album, logic_root, "b_v1")
        _dump({"threads": [THREAD]}, album / "archive_keep.yml")
        folders = classify(logic_root, album)
        assert {f.classification for f in folders} == {ArchiveClassification.KEPT}

    def test_missing_keep_list_and_sides(self, env):
        album, logic_root, _ = env
        _make_song(album, logic_root, "a_v1")
        folders = classify(logic_root, album)
        assert folders[0].classification == ArchiveClassification.CANDIDATE

    def test_placed_matches_on_slug_across_threads(self, env):
        album, logic_root, _ = env
        _make_song(album, logic_root, "shared_v1")
        _place(album, "some-other-thread__shared_v1")
        folders = classify(logic_root, album)
        assert folders[0].classification == ArchiveClassification.PLACED


# ---------------------------------------------------------------------------
# Verified move
# ---------------------------------------------------------------------------


class TestArchiveSong:
    def test_moves_with_thread_hierarchy_and_records_manifest(self, env):
        album, logic_root, archive_root = env
        _, song_dir = _make_song(album, logic_root, "cand_v1")
        folder = classify(logic_root, album)[0]

        dst = archive_song(folder, archive_root, album)

        assert dst == archive_root / THREAD / song_dir.name
        assert (dst / "MIDI" / "chords" / "a.mid").read_bytes() == b"MThd"
        assert not song_dir.exists()
        entry = load_manifest(album).songs[f"{THREAD}__cand_v1"]
        assert entry.status == ArchiveStatus.ARCHIVED
        assert entry.file_count == 3
        assert entry.dest_path == str(dst)

    def test_rewrites_composition_and_song_context(self, env):
        album, logic_root, archive_root = env
        title = "Cand V1"
        old = logic_root / THREAD / f"{title} (cand_v1)"
        prod, song_dir = _make_song(
            album,
            logic_root,
            "cand_v1",
            extra_ctx={"suite_logic_path": str(old / f"{title}.logicx"), "bpm": 90},
        )
        dst = archive_song(classify(logic_root, album)[0], archive_root, album)

        assert _load(dst / "composition.yml")["logic_project_path"] == str(dst)
        ctx = _load(prod / "song_context.yml")
        assert ctx["suite_logic_path"] == str(dst / f"{title}.logicx")
        assert ctx["bpm"] == 90

    def test_checksum_mismatch_leaves_source_untouched(self, env):
        album, logic_root, archive_root = env
        prod, song_dir = _make_song(album, logic_root, "cand_v1")
        ctx_before = (prod / "song_context.yml").read_text()
        folder = classify(logic_root, album)[0]

        real_sha = logic_archive._sha256

        def corrupt(path):
            digest = real_sha(path)
            return "bad" if ".partial" in str(path) else digest

        with patch.object(logic_archive, "_sha256", side_effect=corrupt):
            with pytest.raises(ArchiveVerificationError):
                archive_song(folder, archive_root, album)

        assert (song_dir / "MIDI" / "chords" / "a.mid").exists()
        assert not (archive_root / THREAD).exists()
        assert (prod / "song_context.yml").read_text() == ctx_before
        assert load_manifest(album).songs == {}

    def test_destination_conflict_skips(self, env):
        album, logic_root, archive_root = env
        _, song_dir = _make_song(album, logic_root, "cand_v1")
        existing = archive_root / THREAD / song_dir.name
        existing.mkdir(parents=True)
        (existing / "keep.txt").write_text("x")

        with pytest.raises(FileExistsError):
            archive_song(classify(logic_root, album)[0], archive_root, album)
        assert song_dir.exists()
        assert (existing / "keep.txt").read_text() == "x"

    def test_empty_thread_dir_removed_ignoring_ds_store(self, env):
        album, logic_root, archive_root = env
        _make_song(album, logic_root, "cand_v1")
        (logic_root / THREAD / ".DS_Store").write_bytes(b"x")
        archive_song(classify(logic_root, album)[0], archive_root, album)
        assert not (logic_root / THREAD).exists()

    def test_thread_dir_with_other_content_retained(self, env):
        album, logic_root, archive_root = env
        _make_song(album, logic_root, "cand_v1")
        _make_song(album, logic_root, "placed_v1")
        _place(album, f"{THREAD}__placed_v1")
        folder = _by_slug(classify(logic_root, album), "cand_v1")
        archive_song(folder, archive_root, album)
        assert (logic_root / THREAD).is_dir()


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------


class TestRestore:
    def test_round_trip_recreates_thread_dir(self, env):
        album, logic_root, archive_root = env
        prod, song_dir = _make_song(
            album,
            logic_root,
            "cand_v1",
            extra_ctx={"suite_logic_path": "placeholder"},
        )
        archive_song(classify(logic_root, album)[0], archive_root, album)
        assert not (logic_root / THREAD).exists()

        restored = restore_song(f"{THREAD}__cand_v1", album)

        assert restored == song_dir
        assert (song_dir / "MIDI" / "chords" / "a.mid").exists()
        assert _load(song_dir / "composition.yml")["logic_project_path"] == str(
            song_dir
        )
        assert not (archive_root / THREAD).exists()
        entry = load_manifest(album).songs[f"{THREAD}__cand_v1"]
        assert entry.status == ArchiveStatus.RESTORED

    def test_restore_unknown_song_raises(self, env):
        album, _, _ = env
        with pytest.raises(KeyError):
            restore_song(f"{THREAD}__nope", album)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def _snapshot(self, *roots):
        return {
            str(p): (p.stat().st_mtime_ns if p.is_file() else None)
            for root in roots
            for p in root.rglob("*")
        }

    def test_dry_run_makes_no_changes(self, env, capsys):
        album, logic_root, archive_root = env
        _make_song(album, logic_root, "cand_v1")
        before = self._snapshot(album, logic_root, archive_root)
        assert main(["--album-dir", str(album)]) == 0
        assert self._snapshot(album, logic_root, archive_root) == before
        assert "Dry run" in capsys.readouterr().out

    def test_execute_skips_unmatched_without_flag(self, env):
        album, logic_root, archive_root = env
        _make_song(album, logic_root, "cand_v1")
        orphan = logic_root / THREAD / "Orphan (gone_v1)"
        orphan.mkdir()
        assert main(["--album-dir", str(album), "--execute"]) == 0
        assert orphan.exists()
        assert (archive_root / THREAD / "Cand V1 (cand_v1)").is_dir()

    def test_execute_only_and_limit(self, env):
        album, logic_root, archive_root = env
        _make_song(album, logic_root, "a_v1")
        _make_song(album, logic_root, "b_v1")
        main(["--album-dir", str(album), "--execute", "--only", f"{THREAD}__b_v1"])
        assert (archive_root / THREAD / "B V1 (b_v1)").is_dir()
        assert (logic_root / THREAD / "A V1 (a_v1)").is_dir()

    def test_execute_missing_archive_root_raises(self, env):
        album, logic_root, archive_root = env
        _make_song(album, logic_root, "cand_v1")
        archive_root.rmdir()
        with pytest.raises(EnvironmentError):
            main(["--album-dir", str(album), "--execute"])
        assert (logic_root / THREAD / "Cand V1 (cand_v1)").is_dir()


# ---------------------------------------------------------------------------
# Archive-aware resolution (logic_handoff)
# ---------------------------------------------------------------------------


class TestResolution:
    def test_falls_back_to_archive(self, env):
        album, logic_root, archive_root = env
        prod, _ = _make_song(album, logic_root, "cand_v1")
        dst = archive_song(classify(logic_root, album)[0], archive_root, album)
        assert resolve_song_dir(prod) == dst

    def test_rehandoff_syncs_into_archive_without_duplicate(self, env):
        album, logic_root, archive_root = env
        prod, song_dir = _make_song(album, logic_root, "cand_v1")
        dst = archive_song(classify(logic_root, album)[0], archive_root, album)
        (prod / "chords" / "approved").mkdir(parents=True)
        (prod / "chords" / "approved" / "new.mid").write_bytes(b"MThd")

        assert handoff(prod) == dst
        assert (dst / "MIDI" / "chords" / "new.mid").exists()
        assert not song_dir.exists()

    def test_offline_archive_raises_and_scaffolds_nothing(self, env):
        album, logic_root, archive_root = env
        prod, song_dir = _make_song(album, logic_root, "cand_v1")
        archive_song(classify(logic_root, album)[0], archive_root, album)

        with patch.dict(
            os.environ, {"LOGIC_ARCHIVE_DIR": str(archive_root.parent / "Unmounted")}
        ):
            with pytest.raises(LogicArchiveOfflineError):
                resolve_song_dir(prod)
            with pytest.raises(LogicArchiveOfflineError):
                handoff(prod)
        assert not song_dir.exists()

    def test_unchanged_when_archive_unset(self, env):
        album, logic_root, _ = env
        prod = album / THREAD / "production" / "fresh_v1"
        _dump({"title": "Fresh", "thread": THREAD}, prod / "song_context.yml")
        env_no_archive = {
            k: v for k, v in os.environ.items() if k != "LOGIC_ARCHIVE_DIR"
        }
        with patch.dict(os.environ, env_no_archive, clear=True):
            assert resolve_song_dir(prod) == logic_root / THREAD / "Fresh (fresh_v1)"


class TestUnicodeNormalisation:
    def test_verify_copy_treats_nfc_and_nfd_names_as_equal(self, tmp_path):
        """HFS+ stores names as NFD; APFS keeps NFC. Same file, different bytes."""
        name = "Séance cuōliú.wav"
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        (src / unicodedata.normalize("NFC", name)).write_bytes(b"audio")
        (dst / unicodedata.normalize("NFD", name)).write_bytes(b"audio")

        assert verify_copy(src, dst) == (1, 5)


class TestStaleCompositionPath:
    def test_stale_logic_project_path_is_corrected(self, env):
        """Older handoffs stored the folder name without its (slug) suffix."""
        album, logic_root, archive_root = env
        _, song_dir = _make_song(album, logic_root, "cand_v1")
        _dump(
            {"logic_project_path": str(logic_root / THREAD / "Cand V1")},
            song_dir / "composition.yml",
        )
        dst = archive_song(classify(logic_root, album)[0], archive_root, album)
        assert _load(dst / "composition.yml")["logic_project_path"] == str(dst)
