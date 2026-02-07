"""Tests for generate_negative_constraints utility."""

from pathlib import Path

import yaml

from app.util.generate_negative_constraints import (
    analyze_bpm,
    analyze_concepts,
    analyze_keys,
    collect_titles,
    format_for_prompt,
    generate_constraints,
    normalize_key,
    write_constraints,
)


# --- Helpers ---


def make_index(tmp_path: Path, threads: list[dict]) -> Path:
    """Write a mock index.yml and return its path."""
    index_path = tmp_path / "index.yml"
    with open(index_path, "w") as f:
        yaml.dump({"thread_count": len(threads), "threads": threads}, f)
    return index_path


def make_thread(
    title: str = "Test Song",
    key: str = "C major",
    bpm: int = 96,
    concept: str = "A test concept",
    color: str = "White",
) -> dict:
    """Create a mock thread metadata dict."""
    return {
        "title": title,
        "key": key,
        "bpm": bpm,
        "concept": concept,
        "rainbow_color": color,
        "thread_id": "test-thread-id",
        "directory": "white-test-song",
        "iteration_count": 17,
    }


# --- Unit tests ---


class TestNormalizeKey:
    def test_basic_key(self):
        assert normalize_key("C major") == "C major"

    def test_key_with_parenthetical(self):
        assert (
            normalize_key("C major (resolving through complete chromatic cycle)")
            == "C major"
        )

    def test_chromatic_typo(self):
        assert normalize_key("C hromatic Complete") == "C major"

    def test_all_keys_typo(self):
        assert normalize_key("A ll Keys (Chromatic Convergence)") == "All Keys"

    def test_sharp_minor(self):
        assert normalize_key("F# minor") == "F# minor"

    def test_empty(self):
        assert normalize_key("") == "unknown"


class TestAnalyzeKeys:
    def test_detects_dominant_key(self):
        threads = [make_thread(key="C major")] * 12 + [make_thread(key="F# minor")] * 4
        result = analyze_keys(threads)
        assert len(result["clusters"]) >= 1
        assert result["clusters"][0]["key"] == "C major"
        assert result["clusters"][0]["severity"] == "exclude"  # 75% > 50%

    def test_entropy_low_for_concentrated(self):
        threads = [make_thread(key="C major")] * 20
        result = analyze_keys(threads)
        assert result["entropy"] == 0.0  # All same key

    def test_entropy_high_for_diverse(self):
        keys = [
            "C major",
            "D minor",
            "E major",
            "F minor",
            "G major",
            "A minor",
            "Bb major",
            "F# minor",
        ]
        threads = [make_thread(key=k) for k in keys]
        result = analyze_keys(threads)
        assert result["entropy"] == 3.0  # log2(8) = 3.0

    def test_empty_threads(self):
        result = analyze_keys([])
        assert result["entropy"] == 0.0
        assert result["clusters"] == []

    def test_no_cluster_when_diverse(self):
        keys = ["C", "D", "E", "F", "G", "A", "Bb", "F#", "Ab", "Eb"]
        threads = [make_thread(key=k) for k in keys]
        result = analyze_keys(threads)
        assert len(result["clusters"]) == 0


class TestAnalyzeBpm:
    def test_detects_cluster(self):
        # 10 threads at 91-96, 5 at 108
        threads = (
            [make_thread(bpm=91)] * 5
            + [make_thread(bpm=96)] * 5
            + [make_thread(bpm=108)] * 5
        )
        result = analyze_bpm(threads)
        assert len(result["clusters"]) >= 1

    def test_std_dev_zero_for_same_bpm(self):
        threads = [make_thread(bpm=100)] * 10
        result = analyze_bpm(threads)
        assert result["std_dev"] == 0.0

    def test_std_dev_positive_for_varied(self):
        threads = [make_thread(bpm=60), make_thread(bpm=120)]
        result = analyze_bpm(threads)
        assert result["std_dev"] == 30.0

    def test_empty(self):
        result = analyze_bpm([])
        assert result["std_dev"] == 0.0


class TestAnalyzeConcepts:
    def test_detects_repeated_phrases(self):
        concept = (
            "The transmigration of seven chromatic methodologies into consciousness"
        )
        threads = [make_thread(concept=concept)] * 10
        result = analyze_concepts(threads)
        phrases = [p["phrase"] for p in result["repeated_phrases"]]
        assert "transmigration" in phrases
        assert "seven chromatic methodologies" in phrases

    def test_no_repeated_when_diverse(self):
        threads = [
            make_thread(concept="A song about the ocean and waves"),
            make_thread(concept="Exploring the forest at dawn"),
            make_thread(concept="City lights and midnight jazz"),
        ]
        result = analyze_concepts(threads)
        assert len(result["repeated_phrases"]) == 0

    def test_empty(self):
        result = analyze_concepts([])
        assert result["repeated_phrases"] == []


class TestCollectTitles:
    def test_collects_unique_sorted(self):
        threads = [
            make_thread(title="Zebra Song"),
            make_thread(title="Alpha Song"),
            make_thread(title="Zebra Song"),  # Duplicate
        ]
        titles = collect_titles(threads)
        assert titles == ["Alpha Song", "Zebra Song"]

    def test_empty(self):
        assert collect_titles([]) == []


class TestGenerateConstraints:
    def test_full_generation(self, tmp_path):
        threads = [
            make_thread(
                key="C major",
                bpm=96,
                title=f"Song {i}",
                concept="The transmigration of seven chromatic methodologies",
            )
            for i in range(12)
        ]
        index_path = make_index(tmp_path, threads)
        result = generate_constraints(index_path)

        assert result["thread_count"] == 12
        assert len(result["key_constraints"]) >= 1
        assert len(result["excluded_titles"]) > 0
        assert "key_entropy" in result["diversity_metrics"]
        assert "bpm_std_dev" in result["diversity_metrics"]

    def test_missing_index(self, tmp_path):
        result = generate_constraints(tmp_path / "nonexistent.yml")
        assert "error" in result

    def test_warnings_for_low_entropy(self, tmp_path):
        threads = [make_thread(key="C major", bpm=96)] * 20
        index_path = make_index(tmp_path, threads)
        result = generate_constraints(index_path)
        assert len(result["warnings"]) >= 1
        assert any("entropy" in w.lower() for w in result["warnings"])


class TestFormatForPrompt:
    def test_contains_key_sections(self):
        constraints = {
            "key_constraints": [
                {
                    "key": "C major",
                    "count": 12,
                    "fraction": 0.6,
                    "severity": "exclude",
                    "reason": "12/20 use C major",
                }
            ],
            "bpm_constraints": [
                {
                    "bpm_range": "86-96",
                    "center": 91,
                    "count": 10,
                    "fraction": 0.5,
                    "severity": "avoid",
                    "reason": "10/20 cluster around 91 BPM",
                }
            ],
            "concept_constraints": [
                {
                    "phrase": "transmigration",
                    "count": 7,
                    "fraction": 0.35,
                    "severity": "avoid",
                    "reason": "'transmigration' appears in 7/20 concepts",
                }
            ],
            "excluded_titles": ["Song One", "Song Two"],
            "warnings": ["Key entropy low"],
        }
        text = format_for_prompt(constraints)
        assert "NEGATIVE CONSTRAINTS" in text
        assert "C major" in text
        assert "86-96" in text
        assert "transmigration" in text
        assert "Song One" in text
        assert "DIVERSITY WARNINGS" in text

    def test_empty_constraints(self):
        constraints = {
            "key_constraints": [],
            "bpm_constraints": [],
            "concept_constraints": [],
            "excluded_titles": [],
            "warnings": [],
        }
        text = format_for_prompt(constraints)
        assert "NEGATIVE CONSTRAINTS" in text
        assert "UNEXPLORED" in text


class TestWriteConstraints:
    def test_writes_yaml(self, tmp_path):
        constraints = {"key_constraints": [], "thread_count": 5}
        path = write_constraints(tmp_path / "constraints.yml", constraints)
        assert path.exists()
        with open(path) as f:
            loaded = yaml.safe_load(f)
        assert loaded["thread_count"] == 5

    def test_preserves_manual_overrides(self, tmp_path):
        # Write initial with manual overrides
        path = tmp_path / "constraints.yml"
        initial = {
            "key_constraints": [],
            "manual_overrides": {"force_key": "Eb major"},
        }
        with open(path, "w") as f:
            yaml.dump(initial, f)

        # Regenerate — should preserve manual_overrides
        new_constraints = {"key_constraints": [{"key": "C major"}], "thread_count": 10}
        write_constraints(path, new_constraints)

        with open(path) as f:
            loaded = yaml.safe_load(f)
        assert loaded["manual_overrides"]["force_key"] == "Eb major"
        assert loaded["thread_count"] == 10
