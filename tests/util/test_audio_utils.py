import os
import tempfile
import shutil
from app.util.audio_file_name_utils import rename_mid_audio_extensions

def test_rename_mid_audio_extensions():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        file1 = os.path.join(tmpdir, "test1.mid.wav")
        file2 = os.path.join(tmpdir, "test2.mid.aif")
        file3 = os.path.join(tmpdir, "test3.wav")
        file4 = os.path.join(tmpdir, "test4.aif")
        subdir = os.path.join(tmpdir, "sub")
        os.makedirs(subdir)
        file5 = os.path.join(subdir, "test5.mid.wav")
        file6 = os.path.join(subdir, "test6.mid.aif")
        for f in [file1, file2, file3, file4, file5, file6]:
            with open(f, "w") as fp:
                fp.write("dummy")

        rename_mid_audio_extensions(tmpdir)

        assert not os.path.exists(file1)
        assert os.path.exists(os.path.join(tmpdir, "test1.wav"))
        assert not os.path.exists(file2)
        assert os.path.exists(os.path.join(tmpdir, "test2.aif"))
        assert os.path.exists(file3)
        assert os.path.exists(file4)
        assert not os.path.exists(file5)
        assert os.path.exists(os.path.join(subdir, "test5.wav"))
        assert not os.path.exists(file6)
        assert os.path.exists(os.path.join(subdir, "test6.aif"))
