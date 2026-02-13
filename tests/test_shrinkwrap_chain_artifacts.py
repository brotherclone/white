import os
from pathlib import Path

import pytest

from app.util import shrinkwrap_chain_artifacts as s


def test_is_debug_file_and_chromatic_not_debug():
    # Known debug patterns should be detected (case-insensitive)
    assert s.is_debug_file("white_agent_abc_rebracketing_analysis.md")
    assert s.is_debug_file("some_analysis.JSON")
    assert s.is_debug_file("trace_log.md")

    # CHROMATIC_SYNTHESIS should NOT be treated as debug
    assert not s.is_debug_file("white_agent_abc_CHROMATIC_SYNTHESIS.md")
    assert not s.is_debug_file("white_agent_abc_CHROMATIC_SYNTHESIS.json")


def test_copy_thread_files_lazy_creation_and_debug_handling(tmp_path: Path):
    """Create a fake thread with md/ and wav/ content and ensure:
    - CHROMATIC_SYNTHESIS is copied as a valuable file
    - debug files are skipped when include_debug=False
    - EVP intermediate wav (segment/blended) are skipped
    - no .debug/ is created unless include_debug=True
    - destination subdirs (md/, wav/) are only created when non-debug files copied
    """
    thread = tmp_path / "thread"
    md = thread / "md"
    wav = thread / "wav"
    md.mkdir(parents=True)
    wav.mkdir(parents=True)

    # Valuable/expected files
    chroma = md / "white_agent_uuid_CHROMATIC_SYNTHESIS.md"
    chroma.write_text("chromatic content")

    final_wav = wav / "final_mix.wav"
    final_wav.write_bytes(b"\x01")

    # Debug/intermediate files (should be skipped unless archived)
    doc_syn = md / "white_agent_uuid_document_synthesis.md"
    doc_syn.write_text("doc synth")

    rebrack = md / "white_agent_uuid_rebracketing_analysis.md"
    rebrack.write_text("analysis")

    meta_reb = md / "white_agent_uuid_META_REBRACKETING.md"
    meta_reb.write_text("meta")

    trans_traces = md / "white_agent_uuid_transformation_traces.md"
    trans_traces.write_text("traces")

    # EVP intermediate audio
    evp_seg = wav / "foo_segment_001.wav"
    evp_seg.write_bytes(b"\x00")

    # Destination (first run: exclude debug)
    dest = tmp_path / "out"
    result = s.copy_thread_files(thread, dest, include_debug=False)

    # Expect: chroma + final_mix copied => 2 copied
    assert result["copied"] == 2, f"unexpected copied count: {result}"
    # Expect debug files counted as skipped
    assert result["skipped_debug"] == 4
    # EVP skipped
    assert result["skipped_evp"] == 1

    # Check directories created and files present
    assert (dest / "md").exists() and (dest / "md" / chroma.name).exists()
    assert (dest / "wav").exists() and (dest / "wav" / final_wav.name).exists()
    # No .debug created when not archiving
    assert not (dest / ".debug").exists()

    # Second run: archive debug files
    dest2 = tmp_path / "out2"
    result2 = s.copy_thread_files(thread, dest2, include_debug=True)

    # Now debug files should have been copied in addition to non-debug
    assert result2["skipped_debug"] == 0
    assert result2["skipped_evp"] == 1
    # copied should be chroma + final + 4 debug = 6
    assert result2["copied"] == 6, f"unexpected copied count when archiving: {result2}"

    # .debug should exist and contain the debug files (document_synthesis, rebracketing, META_REBRACKETING, transformation_traces)
    debug_dir = dest2 / ".debug"
    assert debug_dir.exists()
    for dbg in [doc_syn.name, rebrack.name, meta_reb.name, trans_traces.name]:
        assert (debug_dir / dbg).exists(), f"missing debug file in archive: {dbg}"

    # CHROMATIC_SYNTHESIS remains in md/ not in .debug/
    assert (dest2 / "md" / chroma.name).exists()


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
