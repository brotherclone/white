import os
from pathlib import Path
from app.util import audio_file_name_utils as au


def test_rename_mid_audio_extensions(tmp_path):
    root = tmp_path / "music"
    root.mkdir()
    # create files
    f1 = root / "song1.mid.wav"
    f2 = root / "song2.mid.aif"
    f3 = root / "already.wav"
    f1.write_text("x")
    f2.write_text("y")
    f3.write_text("z")

    # run function
    au.rename_mid_audio_extensions(str(root))

    # check renamed
    assert not (root / "song1.mid.wav").exists()
    assert (root / "song1.wav").exists()
    assert not (root / "song2.mid.aif").exists()
    assert (root / "song2.aif").exists()
    # unchanged file remains
    assert (root / "already.wav").exists()

