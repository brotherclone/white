"""Tests for shrinkwrap_chain_artifacts utility."""

from pathlib import Path

import yaml

from white_composition.shrinkwrap_chain_artifacts import (
    clean_filename,
    copy_thread_files,
    find_debug_files,
    generate_directory_name,
    is_debug_file,
    is_evp_intermediate,
    is_uuid,
    parse_thread,
    resolve_collision,
    rewrite_file_name_field,
    scaffold_song_productions,
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
    complete: bool = True,
) -> Path:
    """Create a minimal fake thread directory with song proposal YAML.

    complete=True (default) writes a run_success sentinel so shrinkwrap
    treats this thread as a finished run.  Pass complete=False to simulate
    a crashed/incomplete thread.
    """
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

    if complete:
        (thread_dir / "run_success").touch()

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
        # CHROMATIC_SYNTHESIS is a final synthesis output and should NOT be treated as debug
        assert is_debug_file("white_agent_final_CHROMATIC_SYNTHESIS.md") is False

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

    def test_string_rainbow_color(self, tmp_path):
        thread_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        thread_dir = tmp_path / thread_id
        yml_dir = thread_dir / "yml"
        yml_dir.mkdir(parents=True)
        proposal = {
            "iterations": [
                {
                    "title": "Flat Color Song",
                    "bpm": 95,
                    "key": "D minor",
                    "rainbow_color": "White",
                    "concept": "plain string color",
                    "mood": [],
                    "genres": [],
                    "agent_name": "W",
                }
            ]
        }
        with open(yml_dir / f"all_song_proposals_{thread_id}.yml", "w") as f:
            yaml.dump(proposal, f)
        result = parse_thread(thread_dir)
        assert result is not None
        assert result["rainbow_color"] == "White"
        assert result["mnemonic"] == "?"

    def test_keysig_dict_key(self, tmp_path):
        # KeySignature.model_dump(mode="json") shape stored in proposals
        thread_id = "bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
        thread_dir = tmp_path / thread_id
        yml_dir = thread_dir / "yml"
        yml_dir.mkdir(parents=True)
        proposal = {
            "iterations": [
                {
                    "title": "Dict Key Song",
                    "bpm": 110,
                    "key": {
                        "note": {"pitch_name": "F", "accidental": "sharp"},
                        "mode": {"name": "minor", "intervals": [2, 1, 2, 2, 1, 2, 2]},
                    },
                    "rainbow_color": {
                        "color_name": "Blue",
                        "mnemonic_character_value": "B",
                    },
                    "concept": "test",
                    "mood": [],
                    "genres": [],
                    "agent_name": "B",
                }
            ]
        }
        with open(yml_dir / f"all_song_proposals_{thread_id}.yml", "w") as f:
            yaml.dump(proposal, f)
        result = parse_thread(thread_dir)
        assert result is not None
        assert result["key"] == "F# minor"

    def test_missing_key_fields_returns_unknown(self, tmp_path):
        thread_id = "cccccccc-dddd-eeee-ffff-aaaaaaaaaaaa"
        thread_dir = tmp_path / thread_id
        yml_dir = thread_dir / "yml"
        yml_dir.mkdir(parents=True)
        proposal = {
            "iterations": [
                {
                    "title": "No Key Song",
                    "bpm": 100,
                    "key": {},
                    "rainbow_color": {
                        "color_name": "Red",
                        "mnemonic_character_value": "R",
                    },
                    "concept": "test",
                    "mood": [],
                    "genres": [],
                    "agent_name": "R",
                }
            ]
        }
        with open(yml_dir / f"all_song_proposals_{thread_id}.yml", "w") as f:
            yaml.dump(proposal, f)
        result = parse_thread(thread_dir)
        assert result is not None
        assert result["key"] == "unknown"


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
        add_debug_files(thread_dir, count=3)
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
        add_evp_intermediates(source, count=3)  # 3 segments + 1 blended = 4
        dest = tmp_path / "dest"
        dest.mkdir()

        counts = copy_thread_files(source, dest, include_debug=False)
        assert counts["skipped_evp"] == 4
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

        output = tmp_path / "shrink_wrapped"
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

        output = tmp_path / "shrink_wrapped"
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

        output = tmp_path / "shrink_wrapped"
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

        output = tmp_path / "shrink_wrapped"
        result1 = shrinkwrap(artifacts, output)
        assert result1["processed"] == 1

        result2 = shrinkwrap(artifacts, output)
        assert result2["processed"] == 0  # Skipped on re-run

    def test_thread_filter(self, tmp_path):
        artifacts = tmp_path / "chain_artifacts"
        target_id = "aaaaaaaa-1111-2222-3333-444444444444"
        make_thread(artifacts, target_id, title="Target")
        make_thread(artifacts, "bbbbbbbb-1111-2222-3333-444444444444", title="Other")

        output = tmp_path / "shrink_wrapped"
        result = shrinkwrap(artifacts, output, thread_filter=target_id)

        assert result["processed"] == 1
        assert (output / "white-target").is_dir()
        assert not (output / "white-other").exists()

    def test_missing_artifacts_dir(self, tmp_path):
        result = shrinkwrap(tmp_path / "nonexistent", tmp_path / "out")
        assert result["processed"] == 0


class TestCleanFilename:
    """Task 4.1 — one test per cleaning rule plus the pass-through."""

    def test_uuid_char_prefix(self):
        assert (
            clean_filename(
                "a56f0abe-663e-4763-b40f-dac3c936aa02_g_arbitrarys_survey.md"
            )
            == "arbitrarys_survey.md"
        )

    def test_uuid_char_prefix_html(self):
        assert (
            clean_filename(
                "5546ede0-e6e9-4bea-bf3a-9aac594c0fa2_t_UNKNOWN_ARTIFACT_NAME.html"
            )
            == "UNKNOWN_ARTIFACT_NAME.html"
        )

    def test_white_agent_prefix(self):
        assert (
            clean_filename(
                "white_agent_12c27cb8-d3d8-4513-bc0e-57b8f4449222_AGENT_VOICES.md"
            )
            == "agent_voices.md"
        )

    def test_white_agent_prefix_chromatic(self):
        assert (
            clean_filename(
                "white_agent_12c27cb8-d3d8-4513-bc0e-57b8f4449222_CHROMATIC_SYNTHESIS.md"
            )
            == "chromatic_synthesis.md"
        )

    def test_all_song_proposals(self):
        assert (
            clean_filename(
                "all_song_proposals_12c27cb8-d3d8-4513-bc0e-57b8f4449222.yml"
            )
            == "all_song_proposals.yml"
        )

    def test_all_song_proposals_md(self):
        assert (
            clean_filename("all_song_proposals_12c27cb8-d3d8-4513-bc0e-57b8f4449222.md")
            == "all_song_proposals.md"
        )

    def test_song_proposal_color_hex(self):
        assert (
            clean_filename(
                "song_proposal_Black (0x231f20)_neural_network_incarnation_v2.yml"
            )
            == "neural_network_incarnation_v2.yml"
        )

    def test_song_proposal_single_char(self):
        assert (
            clean_filename(
                "song_proposal_Y_unstable_pantry_crystalline_collapse_v1.yml"
            )
            == "unstable_pantry_crystalline_collapse_v1.yml"
        )

    def test_song_proposal_word_color(self):
        assert (
            clean_filename("song_proposal_indigo_indigo_proposal_1770995946421.yml")
            == "indigo_proposal_1770995946421.yml"
        )

    def test_no_match_passthrough(self):
        assert clean_filename("agent_voices.md") == "agent_voices.md"

    def test_no_match_regular_yml(self):
        assert clean_filename("newspaper_article.yml") == "newspaper_article.yml"


class TestResolveCollision:
    """Task 4.2 — collision suffix logic."""

    def test_no_collision(self):
        assert resolve_collision("foo.md", set()) == "foo.md"

    def test_first_collision(self):
        assert resolve_collision("foo.md", {"foo.md"}) == "foo_2.md"

    def test_second_collision(self):
        assert resolve_collision("foo.md", {"foo.md", "foo_2.md"}) == "foo_3.md"

    def test_no_extension(self):
        assert resolve_collision("readme", {"readme"}) == "readme_2"

    def test_dotfile(self):
        # Hidden files have no suffix in pathlib; suffix appended after stem
        assert resolve_collision(".gitignore", {".gitignore"}) == ".gitignore_2"


class TestRewriteFileNameField:
    def test_rewrites_yml(self, tmp_path):
        f = tmp_path / "test.yml"
        f.write_text("artifact_id: abc\nfile_name: old_name.yml\ntitle: test\n")
        rewrite_file_name_field(f, "new_name.yml")
        assert "file_name: new_name.yml" in f.read_text()

    def test_rewrites_md_yaml_body(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("artifact_id: abc\nfile_name: old_name.md\ncontent: hello\n")
        rewrite_file_name_field(f, "new_name.md")
        assert "file_name: new_name.md" in f.read_text()

    def test_no_field_unchanged(self, tmp_path):
        original = "title: test\nconcept: something\n"
        f = tmp_path / "test.yml"
        f.write_text(original)
        rewrite_file_name_field(f, "whatever.yml")
        assert f.read_text() == original

    def test_missing_file_no_error(self, tmp_path):
        # Should not raise
        rewrite_file_name_field(tmp_path / "nonexistent.yml", "x.yml")


class TestCleanFilenameIntegration:
    """Task 4.3 — shrinkwrap integration: clean names and file_name field rewrite."""

    def test_files_have_clean_names(self, tmp_path):
        thread_id = "c59be431-6527-4424-83fd-4dea6a83edf5"
        artifacts = tmp_path / "chain_artifacts"
        thread_dir = make_thread(artifacts, thread_id, title="Clean Song")

        # Add files with raw UUID-prefixed names
        yml_dir = thread_dir / "yml"
        yml_dir.mkdir(exist_ok=True)
        raw_yml = yml_dir / "03e1727a-0e7f-4624-b049-efdd817b08f8_r_bandwidth_wars.yml"
        raw_yml.write_text(
            "file_name: 03e1727a-0e7f-4624-b049-efdd817b08f8_r_bandwidth_wars.yml\ntitle: test\n"
        )

        md_dir = thread_dir / "md"
        md_dir.mkdir(exist_ok=True)
        raw_md = md_dir / f"white_agent_{thread_id}_CHROMATIC_SYNTHESIS.md"
        raw_md.write_text(
            f"file_name: white_agent_{thread_id}_CHROMATIC_SYNTHESIS.md\ncontent: stuff\n"
        )

        output = tmp_path / "shrink_wrapped"
        output.mkdir()
        shrinkwrap_thread(thread_dir, output, set())

        out_dir = output / "white-clean-song"

        # Raw UUID prefix stripped
        assert (out_dir / "yml" / "bandwidth_wars.yml").exists()
        assert not (out_dir / "yml" / raw_yml.name).exists()

        # white_agent prefix stripped and lowercased
        assert (out_dir / "md" / "chromatic_synthesis.md").exists()
        assert not (out_dir / "md" / raw_md.name).exists()

        # file_name field rewritten inside the yml
        text = (out_dir / "yml" / "bandwidth_wars.yml").read_text()
        assert "file_name: bandwidth_wars.yml" in text

        # file_name field rewritten inside the md
        text = (out_dir / "md" / "chromatic_synthesis.md").read_text()
        assert "file_name: chromatic_synthesis.md" in text

    def test_collision_produces_numbered_suffix(self, tmp_path):
        thread_id = "c59be431-6527-4424-83fd-4dea6a83edf5"
        artifacts = tmp_path / "chain_artifacts"
        thread_dir = make_thread(artifacts, thread_id, title="Collision Song")

        html_dir = thread_dir / "html"
        html_dir.mkdir()
        # Two files that both clean to character_sheet.html
        (
            html_dir / "aaaaaaaa-1111-2222-3333-444444444401_y_character_sheet.html"
        ).write_text("sheet 1")
        (
            html_dir / "aaaaaaaa-1111-2222-3333-444444444402_y_character_sheet.html"
        ).write_text("sheet 2")

        output = tmp_path / "shrink_wrapped"
        output.mkdir()
        shrinkwrap_thread(thread_dir, output, set())

        out_html = output / "white-collision-song" / "html"
        assert (out_html / "character_sheet.html").exists()
        assert (out_html / "character_sheet_2.html").exists()


# ---------------------------------------------------------------------------
# scaffold_song_productions
# ---------------------------------------------------------------------------


def _write_proposal(yml_dir: Path, name: str, extra: dict | None = None) -> Path:
    data = {"title": name, "bpm": 120, "key": "C major", "rainbow_color": "Red"}
    if extra:
        data.update(extra)
    path = yml_dir / f"{name}.yml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


class TestScaffoldSongProductions:
    def test_creates_production_dir_and_manifest(self, tmp_path):
        yml_dir = tmp_path / "yml"
        yml_dir.mkdir()
        _write_proposal(yml_dir, "coral_fever_requiem_v1")

        slugs = scaffold_song_productions(tmp_path, yml_dir)

        assert slugs == ["coral_fever_requiem_v1"]
        manifest = (
            tmp_path
            / "production"
            / "coral_fever_requiem_v1"
            / "manifest_bootstrap.yml"
        )
        assert manifest.exists()
        data = yaml.safe_load(manifest.read_text())
        assert data["bpm"] == 120
        assert data["key"] == "C major"
        assert data["rainbow_color"] == "Red"
        assert data["title"] == "coral_fever_requiem_v1"

    def test_rainbow_color_dict_normalized_to_color_name(self, tmp_path):
        yml_dir = tmp_path / "yml"
        yml_dir.mkdir()
        data = {
            "title": "test",
            "bpm": 120,
            "key": "C major",
            "rainbow_color": {"color_name": "Violet", "hex_value": 123},
        }
        import yaml as _yaml

        with open(yml_dir / "dict_color_song_v1.yml", "w") as f:
            _yaml.dump(data, f)

        scaffold_song_productions(tmp_path, yml_dir)
        manifest = (
            tmp_path / "production" / "dict_color_song_v1" / "manifest_bootstrap.yml"
        )
        loaded = _yaml.safe_load(manifest.read_text())
        assert loaded["rainbow_color"] == "Violet"

    def test_skips_evp_yml(self, tmp_path):
        yml_dir = tmp_path / "yml"
        yml_dir.mkdir()
        evp = yml_dir / "evp.yml"
        evp.write_text("bpm: 120\nkey: C major\nrainbow_color: Red\n")

        slugs = scaffold_song_productions(tmp_path, yml_dir)
        assert slugs == []
        assert not (tmp_path / "production" / "evp").exists()

    def test_skips_all_song_proposals_yml(self, tmp_path):
        yml_dir = tmp_path / "yml"
        yml_dir.mkdir()
        asp = yml_dir / "all_song_proposals.yml"
        asp.write_text("bpm: 120\nkey: C major\nrainbow_color: Red\n")

        slugs = scaffold_song_productions(tmp_path, yml_dir)
        assert slugs == []

    def test_skips_file_missing_required_keys(self, tmp_path):
        yml_dir = tmp_path / "yml"
        yml_dir.mkdir()
        (yml_dir / "no_bpm.yml").write_text("key: C major\nrainbow_color: Red\n")
        (yml_dir / "no_key.yml").write_text("bpm: 120\nrainbow_color: Red\n")
        (yml_dir / "no_color.yml").write_text("bpm: 120\nkey: C major\n")

        slugs = scaffold_song_productions(tmp_path, yml_dir)
        assert slugs == []

    def test_idempotent_does_not_overwrite(self, tmp_path):
        yml_dir = tmp_path / "yml"
        yml_dir.mkdir()
        _write_proposal(yml_dir, "my_song_v1")
        scaffold_song_productions(tmp_path, yml_dir)

        manifest = tmp_path / "production" / "my_song_v1" / "manifest_bootstrap.yml"
        manifest.write_text("title: overwritten\n")

        second_run = scaffold_song_productions(tmp_path, yml_dir)
        assert manifest.read_text() == "title: overwritten\n"
        assert second_run == []  # nothing newly created

    def test_singer_field_included_when_present(self, tmp_path):
        yml_dir = tmp_path / "yml"
        yml_dir.mkdir()
        _write_proposal(yml_dir, "song_with_singer", extra={"singer": "Shirley"})

        scaffold_song_productions(tmp_path, yml_dir)
        manifest = (
            tmp_path / "production" / "song_with_singer" / "manifest_bootstrap.yml"
        )
        data = yaml.safe_load(manifest.read_text())
        assert data["singer"] == "Shirley"

    def test_singer_null_when_absent(self, tmp_path):
        yml_dir = tmp_path / "yml"
        yml_dir.mkdir()
        _write_proposal(yml_dir, "no_singer_song")

        scaffold_song_productions(tmp_path, yml_dir)
        manifest = tmp_path / "production" / "no_singer_song" / "manifest_bootstrap.yml"
        data = yaml.safe_load(manifest.read_text())
        assert data["singer"] is None

    def test_missing_yml_dir_returns_empty(self, tmp_path):
        slugs = scaffold_song_productions(tmp_path, tmp_path / "yml")
        assert slugs == []

    def test_multiple_proposals_all_scaffolded(self, tmp_path):
        yml_dir = tmp_path / "yml"
        yml_dir.mkdir()
        _write_proposal(yml_dir, "song_a_v1")
        _write_proposal(yml_dir, "song_b_v1")

        slugs = scaffold_song_productions(tmp_path, yml_dir)
        assert sorted(slugs) == ["song_a_v1", "song_b_v1"]
        assert (
            tmp_path / "production" / "song_a_v1" / "manifest_bootstrap.yml"
        ).exists()
        assert (
            tmp_path / "production" / "song_b_v1" / "manifest_bootstrap.yml"
        ).exists()

    def test_sounds_like_written_to_manifest(self, tmp_path):
        yml_dir = tmp_path / "yml"
        yml_dir.mkdir()
        _write_proposal(
            yml_dir,
            "song_with_refs",
            extra={"sounds_like": ["Seefeel", "Boards of Canada"]},
        )

        scaffold_song_productions(tmp_path, yml_dir)
        manifest = tmp_path / "production" / "song_with_refs" / "manifest_bootstrap.yml"
        data = yaml.safe_load(manifest.read_text())
        assert data["sounds_like"] == ["Seefeel", "Boards of Canada"]

    def test_sounds_like_defaults_to_empty_list(self, tmp_path):
        yml_dir = tmp_path / "yml"
        yml_dir.mkdir()
        _write_proposal(yml_dir, "song_no_refs")

        scaffold_song_productions(tmp_path, yml_dir)
        manifest = tmp_path / "production" / "song_no_refs" / "manifest_bootstrap.yml"
        data = yaml.safe_load(manifest.read_text())
        assert data["sounds_like"] == []


def make_crashed_thread(tmp_path: Path, thread_id: str) -> Path:
    """Create a thread dir with no valid proposal YAML — simulates a mid-run crash."""
    thread_dir = tmp_path / thread_id
    thread_dir.mkdir(parents=True)
    (thread_dir / "partial_state.json").write_text("{}")
    return thread_dir


class TestShrinkwrapSentinel:
    """Tests for run_success sentinel filtering in shrinkwrap()."""

    def test_complete_thread_is_processed(self, tmp_path):
        artifacts = tmp_path / "chain_artifacts"
        make_thread(
            artifacts, "aaaaaaaa-1111-2222-3333-444444444444", title="Good Song"
        )

        output = tmp_path / "out"
        result = shrinkwrap(artifacts, output)

        assert result["processed"] == 1
        assert result["deleted"] == 0
        assert (output / "white-good-song").is_dir()

    def test_crashed_thread_is_deleted(self, tmp_path):
        """Unparseable thread with no sentinel is deleted."""
        artifacts = tmp_path / "chain_artifacts"
        thread_dir = make_crashed_thread(
            artifacts, "bbbbbbbb-1111-2222-3333-444444444444"
        )

        output = tmp_path / "out"
        result = shrinkwrap(artifacts, output)

        assert result["processed"] == 0
        assert result["deleted"] == 1
        assert not thread_dir.exists()

    def test_crashed_thread_skipped_when_delete_disabled(self, tmp_path):
        """Unparseable thread with no sentinel is left alone when delete_incomplete=False."""
        artifacts = tmp_path / "chain_artifacts"
        thread_dir = make_crashed_thread(
            artifacts, "cccccccc-1111-2222-3333-444444444444"
        )

        output = tmp_path / "out"
        result = shrinkwrap(artifacts, output, delete_incomplete=False)

        assert result["processed"] == 0
        assert result["deleted"] == 0
        assert thread_dir.exists()

    def test_legacy_thread_without_sentinel_is_processed(self, tmp_path):
        """Thread with valid proposals but no sentinel is treated as legacy and processed."""
        artifacts = tmp_path / "chain_artifacts"
        make_thread(
            artifacts,
            "dddddddd-1111-2222-3333-444444444444",
            title="Legacy Song",
            complete=False,
        )

        output = tmp_path / "out"
        result = shrinkwrap(artifacts, output)

        assert result["processed"] == 1
        assert result["deleted"] == 0
        assert (output / "white-legacy-song").is_dir()

    def test_mixed_complete_and_crashed(self, tmp_path):
        artifacts = tmp_path / "chain_artifacts"
        make_thread(artifacts, "aaaaaaaa-1111-2222-3333-444444444444", title="Done")
        crashed = make_crashed_thread(artifacts, "bbbbbbbb-1111-2222-3333-444444444444")

        output = tmp_path / "out"
        result = shrinkwrap(artifacts, output)

        assert result["processed"] == 1
        assert result["deleted"] == 1
        assert (output / "white-done").is_dir()
        assert not crashed.exists()


class TestShrinkwrapCatalogHook:
    def test_catalog_failure_does_not_propagate(self, tmp_path):
        """A catalog update error must not abort shrinkwrap."""
        from unittest.mock import patch

        artifacts = tmp_path / "chain_artifacts"
        output = tmp_path / "out"
        thread_id = "aaaaaaaa-0000-0000-0000-000000000001"
        make_thread(artifacts, thread_id, title="Done", color="White")

        with patch(
            "white_generation.artist_catalog.generate_missing",
            side_effect=RuntimeError("catalog unavailable"),
        ):
            result = shrinkwrap(artifacts, output)

        assert result["processed"] == 1
