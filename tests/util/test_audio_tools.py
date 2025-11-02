"""Unit tests for audio_tools.py prioritized WAV file finder"""
import os
import pytest


def _make_files(tmp_path, names):
    """Helper to create dummy files for testing"""
    for n in names:
        p = tmp_path / n
        p.write_bytes(b"")  # empty file is fine for filename-based tests
    return [str(tmp_path / n) for n in names]


def test_find_wav_files_prioritized_orders_priority_first(tmp_path):
    """Test that files with 'vocal' or 'vox' in name appear first"""
    from app.agents.tools.audio_tools import find_wav_files_prioritized

    names = [
        "a.wav",
        "lead_vox.wav",
        "vocal_1.wav",
        "vox_lead.wav",
        "Vocal_solo.WAV",
        "notwav.txt"
    ]
    created = _make_files(tmp_path, names)

    found = find_wav_files_prioritized(str(tmp_path), prefix=None)

    # All .wav files should be present
    expected_wavs = sorted([p for p in created if p.lower().endswith(".wav")],
                          key=lambda p: os.path.basename(p).lower())

    # Compute expected prioritized-first ordering
    priority_keywords = ["vocal", "vox"]
    priority = [p for p in expected_wavs
                if any(k in os.path.basename(p).lower() for k in priority_keywords)]
    non_priority = [p for p in expected_wavs if p not in priority]
    expected_order = (sorted(priority, key=lambda p: os.path.basename(p).lower()) +
                     sorted(non_priority, key=lambda p: os.path.basename(p).lower()))

    assert found == expected_order

    # Verify priority files come first
    assert len(priority) == 4  # lead_vox, vocal_1, vox_lead, Vocal_solo
    assert all("vocal" in os.path.basename(f).lower() or "vox" in os.path.basename(f).lower()
              for f in found[:4])


def test_find_wav_files_prioritized_prefix_filtering(tmp_path):
    """Test that prefix filtering works correctly"""
    from app.agents.tools.audio_tools import find_wav_files_prioritized

    names = [
        "vocal_a.wav",
        "Vocal_b.wav",
        "lead_file.wav",
        "voice_lead.wav",
        "other.wav"
    ]
    created = _make_files(tmp_path, names)

    # prefix is case-sensitive: only lowercase 'vocal_a.wav' should match prefix 'vocal'
    found_prefix = find_wav_files_prioritized(str(tmp_path), prefix="vocal",
                                              priority_keywords=None)
    assert found_prefix == [str(tmp_path / "vocal_a.wav")]


def test_find_wav_files_prioritized_custom_keywords(tmp_path):
    """Test that custom priority keywords work"""
    from app.agents.tools.audio_tools import find_wav_files_prioritized

    names = [
        "vocal_a.wav",
        "Vocal_b.wav",
        "lead_file.wav",
        "voice_lead.wav",
        "other.wav"
    ]
    created = _make_files(tmp_path, names)

    # custom priority keyword should prioritize 'lead' occurrences
    found_custom = find_wav_files_prioritized(str(tmp_path), prefix=None,
                                              priority_keywords=["lead"])

    # ensure files containing 'lead' come first
    basenames = [os.path.basename(p).lower() for p in found_custom]

    # both 'lead_file.wav' and 'voice_lead.wav' should be before other wavs
    first_priority = [b for b in basenames if "lead" in b]
    assert len(first_priority) == 2

    # Check they come first
    assert "lead" in basenames[0]
    assert "lead" in basenames[1]

    # overall set matches all wavs present
    expected_set = {str(tmp_path / n) for n in names if n.lower().endswith(".wav")}
    assert set(found_custom) == expected_set


def test_find_wav_files_prioritized_empty_directory(tmp_path):
    """Test behavior with empty directory"""
    from app.agents.tools.audio_tools import find_wav_files_prioritized

    found = find_wav_files_prioritized(str(tmp_path))
    assert found == []


def test_find_wav_files_prioritized_no_matches(tmp_path):
    """Test behavior when no priority files match"""
    from app.agents.tools.audio_tools import find_wav_files_prioritized

    names = ["a.wav", "b.wav", "c.wav"]
    _make_files(tmp_path, names)

    found = find_wav_files_prioritized(str(tmp_path))

    # Should return all files in sorted order since none match priority
    expected = sorted([str(tmp_path / n) for n in names],
                     key=lambda p: os.path.basename(p).lower())
    assert found == expected


def test_find_wav_files_prioritized_case_insensitive_priority(tmp_path):
    """Test that priority matching is case-insensitive"""
    from app.agents.tools.audio_tools import find_wav_files_prioritized

    names = [
        "VOCAL_TRACK.wav",
        "VoX_MiX.wav",
        "normal.wav"
    ]
    _make_files(tmp_path, names)

    found = find_wav_files_prioritized(str(tmp_path))

    # First two should be the priority files
    basenames = [os.path.basename(p).lower() for p in found[:2]]
    assert "vocal" in basenames[0] or "vox" in basenames[0]
    assert "vocal" in basenames[1] or "vox" in basenames[1]

