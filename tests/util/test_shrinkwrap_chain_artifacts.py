"""Tests for shrinkwrap_chain_artifacts utility."""

from pathlib import Path

import yaml

from app.util.shrinkwrap_chain_artifacts import (
    copy_thread_files,
    find_debug_files,
    generate_directory_name,
    is_debug_file,
    is_evp_intermediate,
    is_uuid,
    parse_thread,
    shrinkwrap,
    shrinkwrap_thread,
    slugify,
    write_index,
    write_manifest,
)


# --- Helpers ---


def make_thread(
    tmp_path: Path,
    thread_id: str,
    title: str = "Test Song",
    color: str = "White",
    bpm: int = 120,
    key: str = "C major",
    iterations: int = 1,
) -> Path:
    """Create a minimal fake thread directory with song proposal YAML."""
    thread_dir = tmp_path / thread_id
    yml_dir = thread_dir / "yml"
    yml_dir.mkdir(parents=True)

    proposal = {
        "iterations": [
            {
                "title": title,
                "bpm": bpm,
                "key": key,
                "tempo": {"numerator": 4, "denominator": 4},
                "concept": f"A test concept for {title}",
                "rainbow_color": {
                    "color_name": color,
                    "mnemonic_character_value": color[0],
                },
                "mood": ["test"],
                "genres": ["test-genre"],
                "agent_name": "TestAgent",
                "timestamp": "2026-01-01T00:00:00",
            }
            for _ in range(iterations)
        ]
    }

    with open(yml_dir / f"all_song_proposals_{thread_id}.yml", "w") as f:
        yaml.dump(proposal, f)

    return thread_dir


def add_debug_files(thread_dir: Path, count: int = 3) -> list[Path]:
    """Add debug markdown files to a thread directory."""
    md_dir = thread_dir / "md"
    md_dir.mkdir(exist_ok=True)
    files = []
    for i in range(count):
        name = f"white_agent_test_{i}_rebracketing_analysis.md"
        path = md_dir / name
        path.write_text(f"Debug content {i}")
        files.append(path)
    return files


def add_evp_intermediates(thread_dir: Path, count: int = 3) -> list[Path]:
    """Add legacy EVP intermediate WAV files to a thread directory."""
    wav_dir = thread_dir / "wav"
    wav_dir.mkdir(exist_ok=True)
    files = []
    for i in range(count):
        name = f"abcdef12-3456-7890-abcd-ef1234567890_z_segment_{i+1}.wav"
        path = wav_dir / name
        path.write_text(f"fake segment audio {i}")
        files.append(path)
    # Also add a blended file
    blended = wav_dir / "blended_final.wav"
    blended.write_text("fake blended audio")
    files.append(blended)
    return files


def add_content_files(thread_dir: Path) -> list[Path]:
    """Add non-debug content files to a thread directory."""
    files = []
    for subdir_name in ["md", "yml", "wav"]:
        subdir = thread_dir / subdir_name
        subdir.mkdir(exist_ok=True)
        path = subdir / f"content_{subdir_name}.txt"
        path.write_text(f"Content in {subdir_name}")
        files.append(path)
    return files


# --- Unit tests ---


class TestIsUuid:
    def test_valid_uuid(self):
        assert is_uuid("c59be431-6527-4424-83fd-4dea6a83edf5") is True

    def test_invalid_uuid(self):
        assert is_uuid("not-a-uuid") is False

    def test_empty_string(self):
        assert is_uuid("") is False

    def test_slug_name(self):
        assert is_uuid("white-the-prism-protocol") is False


class TestSlugify:
    def test_basic(self):
        assert slugify("The Prism Protocol") == "the-prism-protocol"

    def test_special_characters(self):
        assert slugify("It's A Test!") == "its-a-test"

    def test_multiple_spaces(self):
        assert slugify("Too   Many   Spaces") == "too-many-spaces"

    def test_length_cap(self):
        result = slugify("A" * 100)
        assert len(result) <= 80

    def test_parentheses(self):
        assert slugify("Title (Subtitle)") == "title-subtitle"

    def test_apostrophe_variants(self):
        assert slugify("don\u2019t stop") == "dont-stop"


class TestIsDebugFile:
    def test_rebracketing(self):
        assert is_debug_file("white_agent_red_rebracketing_analysis.md") is True

    def test_document_synthesis(self):
        assert is_debug_file("white_agent_blue_document_synthesis.md") is True

    def test_chromatic_synthesis(self):
        assert is_debug_file("white_agent_final_CHROMATIC_SYNTHESIS.md") is True

    def test_facet_evolution(self):
        assert is_debug_file("white_agent_orange_facet_evolution.md") is True

    def test_transformation_traces(self):
        assert is_debug_file("white_agent_green_transformation_traces.md") is True

    def test_meta_rebracketing(self):
        assert is_debug_file("white_agent_indigo_META_REBRACKETING.md") is True

    def test_non_debug_file(self):
        assert is_debug_file("song_proposal_final.yml") is False

    def test_content_md_file(self):
        assert is_debug_file("interview_with_the_sultan.md") is False


class TestIsEvpIntermediate:
    def test_segment_file(self):
        assert (
            is_evp_intermediate("76b4c959-028f-43df-a3dd-7c107fdac543_z_segment_8.wav")
            is True
        )

    def test_segment_file_different_number(self):
        assert is_evp_intermediate("abc123_z_segment_1.wav") is True

    def test_blended_file(self):
        assert is_evp_intermediate("blended_audio.wav") is True

    def test_mosaic_file_not_intermediate(self):
        assert is_evp_intermediate("mosaic_final.wav") is False

    def test_regular_wav(self):
        assert is_evp_intermediate("song_output.wav") is False

    def test_non_wav(self):
        assert is_evp_intermediate("segment_notes.md") is False


class TestParseThread:
    def test_valid_thread(self, tmp_path):
        thread_dir = make_thread(
            tmp_path,
            "c59be431-6527-4424-83fd-4dea6a83edf5",
            title="Test Song",
            bpm=108,
            key="C major",
        )
        result = parse_thread(thread_dir)
        assert result is not None
        assert result["title"] == "Test Song"
        assert result["bpm"] == 108
        assert result["key"] == "C major"
        assert result["rainbow_color"] == "White"
        assert result["iteration_count"] == 1

    def test_multiple_iterations_uses_last(self, tmp_path):
        thread_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        thread_dir = tmp_path / thread_id
        yml_dir = thread_dir / "yml"
        yml_dir.mkdir(parents=True)

        proposal = {
            "iterations": [
                {
                    "title": "First Draft",
                    "bpm": 90,
                    "key": "A minor",
                    "rainbow_color": {
                        "color_name": "Red",
                        "mnemonic_character_value": "R",
                    },
                    "concept": "early",
                    "mood": [],
                    "genres": [],
                    "agent_name": "A1",
                },
                {
                    "title": "Final Version",
                    "bpm": 120,
                    "key": "C major",
                    "rainbow_color": {
                        "color_name": "White",
                        "mnemonic_character_value": "W",
                    },
                    "concept": "final",
                    "mood": ["done"],
                    "genres": ["pop"],
                    "agent_name": "A2",
                },
            ]
        }
        with open(yml_dir / f"all_song_proposals_{thread_id}.yml", "w") as f:
            yaml.dump(proposal, f)

        result = parse_thread(thread_dir)
        assert result["title"] == "Final Version"
        assert result["bpm"] == 120
        assert result["iteration_count"] == 2

    def test_no_proposals(self, tmp_path):
        thread_dir = tmp_path / "c59be431-6527-4424-83fd-4dea6a83edf5"
        (thread_dir / "yml").mkdir(parents=True)
        result = parse_thread(thread_dir)
        assert result is None

    def test_empty_iterations(self, tmp_path):
        thread_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        thread_dir = tmp_path / thread_id
        yml_dir = thread_dir / "yml"
        yml_dir.mkdir(parents=True)
        proposal = {"iterations": []}
        with open(yml_dir / f"all_song_proposals_{thread_id}.yml", "w") as f:
            yaml.dump(proposal, f)
        result = parse_thread(thread_dir)
        assert result is None


class TestGenerateDirectoryName:
    def test_basic(self):
        meta = {"rainbow_color": "White", "title": "The Prism Protocol"}
        name = generate_directory_name(meta, set())
        assert name == "white-the-prism-protocol"

    def test_collision_handling(self):
        meta = {"rainbow_color": "White", "title": "The Prism Protocol"}
        existing = {"white-the-prism-protocol"}
        name = generate_directory_name(meta, existing)
        assert name == "white-the-prism-protocol-2"

    def test_double_collision(self):
        meta = {"rainbow_color": "White", "title": "Test"}
        existing = {"white-test", "white-test-2"}
        name = generate_directory_name(meta, existing)
        assert name == "white-test-3"


class TestFindDebugFiles:
    def test_finds_debug_files(self, tmp_path):
        thread_dir = make_thread(tmp_path, "c59be431-6527-4424-83fd-4dea6a83edf5")
        found = find_debug_files(thread_dir)
        assert len(found) == 3

    def test_no_md_dir(self, tmp_path):
        thread_dir = make_thread(tmp_path, "c59be431-6527-4424-83fd-4dea6a83edf5")
        found = find_debug_files(thread_dir)
        assert len(found) == 0

    def test_ignores_non_debug(self, tmp_path):
        thread_dir = make_thread(tmp_path, "c59be431-6527-4424-83fd-4dea6a83edf5")
        md_dir = thread_dir / "md"
        md_dir.mkdir(exist_ok=True)
        (md_dir / "interview.md").write_text("Not debug")
        (md_dir / "white_agent_red_rebracketing_analysis.md").write_text("Debug")
        found = find_debug_files(thread_dir)
        assert len(found) == 1


class TestCopyThreadFiles:
    def test_copies_non_debug(self, tmp_path):
        source = make_thread(
            tmp_path / "source", "c59be431-6527-4424-83fd-4dea6a83edf5"
        )
        add_content_files(source)
        add_debug_files(source, count=2)

        dest = tmp_path / "dest"
        dest.mkdir()

        counts = copy_thread_files(source, dest, include_debug=False)
        assert counts["skipped_debug"] == 2
        assert counts["copied"] >= 3  # At least the content files

    def test_archive_mode(self, tmp_path):
        source = make_thread(
            tmp_path / "source", "c59be431-6527-4424-83fd-4dea6a83edf5"
        )
        add_content_files(source)
        add_debug_files(source, count=2)

        dest = tmp_path / "dest"
        dest.mkdir()

        counts = copy_thread_files(source, dest, include_debug=True)
        assert counts["skipped_debug"] == 0
        assert (dest / ".debug").exists()

    def test_skips_evp_intermediates(self, tmp_path):
        source = make_thread(
            tmp_path / "source", "c59be431-6527-4424-83fd-4dea6a83edf5"
        )
        add_content_files(source)
        dest = tmp_path / "dest"
        dest.mkdir()

        counts = copy_thread_files(source, dest, include_debug=False)
        assert counts["skipped_evp"] == 4
        # Verify none of the segment/blended files were copied
        for f in (dest / "wav").iterdir():
            assert "_segment_" not in f.name
            assert not f.name.startswith("blended")

    def test_skips_ds_store(self, tmp_path):
        source = make_thread(
            tmp_path / "source", "c59be431-6527-4424-83fd-4dea6a83edf5"
        )
        md_dir = source / "md"
        md_dir.mkdir(exist_ok=True)
        (md_dir / ".DS_Store").write_text("mac junk")
        (md_dir / "real_file.md").write_text("real content")

        dest = tmp_path / "dest"
        dest.mkdir()

        copy_thread_files(source, dest)
        assert not (dest / "md" / ".DS_Store").exists()
        assert (dest / "md" / "real_file.md").exists()


class TestWriteManifest:
    def test_writes_valid_yaml(self, tmp_path):
        meta = {
            "title": "Test",
            "bpm": 120,
            "key": "C major",
            "tempo": {"numerator": 4, "denominator": 4},
            "concept": "A concept",
            "rainbow_color": "White",
            "mnemonic": "W",
            "mood": ["happy"],
            "genres": ["pop"],
            "agent_name": "TestAgent",
            "iteration_count": 5,
            "thread_id": "abc-123",
            "timestamp": None,
        }
        path = write_manifest(tmp_path, meta)
        assert path.exists()
        with open(path) as f:
            loaded = yaml.safe_load(f)
        assert loaded["title"] == "Test"
        assert loaded["bpm"] == 120
        assert loaded["iteration_count"] == 5


class TestWriteIndex:
    def test_writes_index_sorted(self, tmp_path):
        metadata_list = [
            {
                "title": "Zebra",
                "thread_id": "z",
                "directory_name": "white-zebra",
                "bpm": 100,
                "key": "C",
                "rainbow_color": "White",
                "concept": "z",
                "iteration_count": 1,
            },
            {
                "title": "Alpha",
                "thread_id": "a",
                "directory_name": "white-alpha",
                "bpm": 90,
                "key": "D",
                "rainbow_color": "White",
                "concept": "a",
                "iteration_count": 2,
            },
        ]
        path = write_index(tmp_path, metadata_list)
        with open(path) as f:
            index = yaml.safe_load(f)
        assert index["thread_count"] == 2
        assert index["threads"][0]["title"] == "Alpha"
        assert index["threads"][1]["title"] == "Zebra"


class TestShrinkwrapThread:
    def test_full_thread_processing(self, tmp_path):
        artifacts = tmp_path / "chain_artifacts"
        thread_id = "c59be431-6527-4424-83fd-4dea6a83edf5"
        thread_dir = make_thread(artifacts, thread_id, title="My Song")
        add_content_files(thread_dir)
        add_debug_files(thread_dir)

        output = tmp_path / "shrinkwrapped"
        output.mkdir()

        existing = set()
        result = shrinkwrap_thread(thread_dir, output, existing)

        assert result is not None
        assert result["title"] == "My Song"
        assert result["directory_name"] == "white-my-song"
        assert (output / "white-my-song" / "manifest.yml").exists()
        assert "white-my-song" in existing

    def test_skips_non_uuid(self, tmp_path):
        thread_dir = tmp_path / "not-a-uuid"
        thread_dir.mkdir()
        result = shrinkwrap_thread(thread_dir, tmp_path / "out", set())
        assert result is None

    def test_dry_run(self, tmp_path):
        artifacts = tmp_path / "chain_artifacts"
        thread_id = "c59be431-6527-4424-83fd-4dea6a83edf5"
        make_thread(artifacts, thread_id, title="Dry Run Song")

        output = tmp_path / "shrinkwrapped"
        existing = set()
        result = shrinkwrap_thread(
            artifacts / thread_id, output, existing, dry_run=True
        )

        assert result is not None
        assert not output.exists()  # Nothing written
        assert "white-dry-run-song" in existing  # But name was reserved


class TestShrinkwrap:
    def test_full_pipeline(self, tmp_path):
        artifacts = tmp_path / "chain_artifacts"
        make_thread(
            artifacts, "aaaaaaaa-1111-2222-3333-444444444444", title="Song One", bpm=90
        )
        add_content_files(artifacts / "aaaaaaaa-1111-2222-3333-444444444444")
        make_thread(
            artifacts, "bbbbbbbb-1111-2222-3333-444444444444", title="Song Two", bpm=120
        )
        add_content_files(artifacts / "bbbbbbbb-1111-2222-3333-444444444444")

        output = tmp_path / "shrinkwrapped"
        result = shrinkwrap(artifacts, output)

        assert result["processed"] == 2
        assert result["failed"] == 0
        assert (output / "index.yml").exists()
        assert (output / "white-song-one").is_dir()
        assert (output / "white-song-two").is_dir()

    def test_idempotent(self, tmp_path):
        artifacts = tmp_path / "chain_artifacts"
        make_thread(
            artifacts, "aaaaaaaa-1111-2222-3333-444444444444", title="Idempotent Song"
        )
        add_content_files(artifacts / "aaaaaaaa-1111-2222-3333-444444444444")

        output = tmp_path / "shrinkwrapped"
        result1 = shrinkwrap(artifacts, output)
        assert result1["processed"] == 1

        result2 = shrinkwrap(artifacts, output)
        assert result2["processed"] == 0  # Skipped on re-run

    def test_thread_filter(self, tmp_path):
        artifacts = tmp_path / "chain_artifacts"
        target_id = "aaaaaaaa-1111-2222-3333-444444444444"
        make_thread(artifacts, target_id, title="Target")
        make_thread(artifacts, "bbbbbbbb-1111-2222-3333-444444444444", title="Other")

        output = tmp_path / "shrinkwrapped"
        result = shrinkwrap(artifacts, output, thread_filter=target_id)

        assert result["processed"] == 1
        assert (output / "white-target").is_dir()
        assert not (output / "white-other").exists()

    def test_missing_artifacts_dir(self, tmp_path):
        result = shrinkwrap(tmp_path / "nonexistent", tmp_path / "out")
        assert result["processed"] == 0
