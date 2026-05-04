"""Tests for register_part() in promote_part.py."""

import io
from pathlib import Path

import mido
import pytest
import yaml

from white_composition.promote_part import register_part

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_midi_file(path: Path) -> Path:
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message("note_on", note=60, velocity=64, time=0))
    track.append(mido.Message("note_off", note=60, velocity=64, time=480))
    buf = io.BytesIO()
    mid.save(file=buf)
    path.write_bytes(buf.getvalue())
    return path


def _read_review(phase_dir: Path) -> dict:
    with open(phase_dir / "review.yml") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestRegisterPartHappyPath:
    def test_copies_midi_to_approved_dir(self, tmp_path):
        src = _make_midi_file(tmp_path / "chorus_v1.mid")
        prod = tmp_path / "prod"
        register_part(src, "melody", "chorus", "chorus-nongenv1", prod)
        assert (prod / "melody" / "approved" / "chorus_nongenv1.mid").exists()

    def test_review_yml_entry_fields(self, tmp_path):
        src = _make_midi_file(tmp_path / "chorus_v1.mid")
        prod = tmp_path / "prod"
        register_part(src, "melody", "chorus", "chorus-nongenv1", prod)
        review = _read_review(prod / "melody")
        c = review["candidates"][0]
        assert c["id"] == "hand_chorus_nongenv1"
        assert c["label"] == "chorus_nongenv1"
        assert c["section"] == "chorus"
        assert c["status"] == "approved"
        assert c["generated"] is False
        assert c["scores"] is None
        assert c["rank"] is None

    def test_midi_file_field_points_to_approved_dir(self, tmp_path):
        src = _make_midi_file(tmp_path / "chorus_v1.mid")
        prod = tmp_path / "prod"
        register_part(src, "melody", "chorus", "chorus-label", prod)
        review = _read_review(prod / "melody")
        assert review["candidates"][0]["midi_file"] == "approved/chorus_label.mid"

    def test_creates_phase_and_approved_dirs(self, tmp_path):
        src = _make_midi_file(tmp_path / "part.mid")
        prod = tmp_path / "prod"
        register_part(src, "bass", "verse", "verse-hand", prod)
        assert (prod / "bass" / "approved").is_dir()

    def test_appends_to_existing_review_yml(self, tmp_path):
        prod = tmp_path / "prod"
        phase_dir = prod / "melody"
        phase_dir.mkdir(parents=True)
        with open(phase_dir / "review.yml", "w") as f:
            yaml.dump(
                {
                    "candidates": [
                        {
                            "id": "mel_001",
                            "midi_file": "candidates/mel_001.mid",
                            "status": "pending",
                            "label": "verse",
                            "section": "verse",
                            "scores": {"composite": 0.7},
                            "rank": 1,
                        }
                    ]
                },
                f,
            )
        src = _make_midi_file(tmp_path / "hand.mid")
        register_part(src, "melody", "chorus", "chorus-hand", prod)
        review = _read_review(phase_dir)
        assert len(review["candidates"]) == 2

    def test_label_normalised_to_snake_case(self, tmp_path):
        src = _make_midi_file(tmp_path / "part.mid")
        prod = tmp_path / "prod"
        register_part(src, "drums", "bridge", "My-Bridge Part", prod)
        review = _read_review(prod / "drums")
        assert review["candidates"][0]["label"] == "my_bridge_part"
        assert (prod / "drums" / "approved" / "my_bridge_part.mid").exists()

    def test_returns_entry_dict(self, tmp_path):
        src = _make_midi_file(tmp_path / "part.mid")
        prod = tmp_path / "prod"
        entry = register_part(src, "chords", "verse", "verse-hand", prod)
        assert entry["id"] == "hand_verse_hand"
        assert entry["status"] == "approved"
        assert entry["generated"] is False


# ---------------------------------------------------------------------------
# Corrupt MIDI rejected
# ---------------------------------------------------------------------------


class TestRegisterPartCorruptMidi:
    def test_corrupt_midi_raises_value_error(self, tmp_path):
        bad = tmp_path / "bad.mid"
        bad.write_bytes(b"not a midi file at all")
        prod = tmp_path / "prod"
        with pytest.raises(ValueError, match="Cannot read MIDI file"):
            register_part(bad, "melody", "verse", "verse-hand", prod)

    def test_corrupt_midi_writes_no_files(self, tmp_path):
        bad = tmp_path / "bad.mid"
        bad.write_bytes(b"\x00\x00\x00\x00")
        prod = tmp_path / "prod"
        with pytest.raises(ValueError):
            register_part(bad, "melody", "verse", "verse-hand", prod)
        assert not (prod / "melody" / "approved").exists()


# ---------------------------------------------------------------------------
# Duplicate label prevented
# ---------------------------------------------------------------------------


class TestRegisterPartDuplicateLabel:
    def test_duplicate_approved_label_raises(self, tmp_path):
        src = _make_midi_file(tmp_path / "part.mid")
        prod = tmp_path / "prod"
        register_part(src, "melody", "chorus", "chorus-hand", prod)
        src2 = _make_midi_file(tmp_path / "part2.mid")
        with pytest.raises(ValueError, match="already exists as an approved entry"):
            register_part(src2, "melody", "chorus", "chorus-hand", prod)

    def test_pending_label_does_not_block_registration(self, tmp_path):
        prod = tmp_path / "prod"
        phase_dir = prod / "melody"
        phase_dir.mkdir(parents=True)
        with open(phase_dir / "review.yml", "w") as f:
            yaml.dump(
                {
                    "candidates": [
                        {
                            "id": "mel_001",
                            "midi_file": "candidates/mel_001.mid",
                            "status": "pending",
                            "label": "chorus_hand",
                            "section": "chorus",
                            "scores": None,
                            "rank": None,
                        }
                    ]
                },
                f,
            )
        src = _make_midi_file(tmp_path / "part.mid")
        entry = register_part(src, "melody", "chorus", "chorus-hand", prod)
        assert entry["id"] == "hand_chorus_hand"
