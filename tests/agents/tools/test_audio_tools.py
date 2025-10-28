import io
import os
import numpy as np
import soundfile as sf
import pytest
from app.agents.tools import audio_tools
from app.agents.enums.noise_type import NoiseType
from app.agents.enums.chain_artifact_file_type import ChainArtifactFileType
from app.agents.models.audio_chain_artifact_file import AudioChainArtifactFile
from app.structures.concepts.rainbow_table_color import the_rainbow_table_colors


def make_sine_bytes(duration_sec=0.05, sr=44100):
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 440 * t)
    int16 = (sig * 32767).astype(np.int16)
    return int16.tobytes()


def test_generate_speech_like_noise_length():
    duration = 1.0
    sr = 8000
    result = audio_tools.generate_speech_like_noise(duration, sr)
    assert isinstance(result, bytes)
    assert len(result) == int(duration * sr) * 2  # int16


def test_generate_noise_white_length():
    duration = 0.5
    sr = 16000
    result = audio_tools.generate_noise(duration, NoiseType.WHITE, sample_rate=sr)
    assert isinstance(result, bytes)
    assert len(result) == int(duration * sr) * 2


def test_generate_noise_invalid_type():
    duration = 0.1
    sr = 8000
    result = audio_tools.generate_noise(duration, None, sample_rate=sr)
    assert isinstance(result, bytes)
    assert len(result) == int(duration * sr) * 2


def test_pitch_shift_audio_bytes_identity():
    sr = 8000
    arr = (np.random.rand(sr) * 2 - 1).astype(np.float32)
    arr_bytes = (arr * 32767).astype(np.int16).tobytes()
    shifted = audio_tools.pitch_shift_audio_bytes(arr_bytes, cents=0, sample_rate=sr)
    assert isinstance(shifted, bytes)
    assert len(shifted) == len(arr_bytes)


def test_micro_stutter_audio_bytes():
    sr = 8000
    arr = (np.random.rand(sr) * 2 - 1).astype(np.float32)
    arr_bytes = (arr * 32767).astype(np.int16).tobytes()
    stuttered = audio_tools.micro_stutter_audio_bytes(arr_bytes, stutter_probability=1.0, stutter_length_ms=10, sample_rate=sr)
    assert isinstance(stuttered, bytes)
    assert len(stuttered) >= len(arr_bytes)


def test_gate_audio_bytes():
    sr = 8000
    arr = (np.random.rand(sr) * 2 - 1).astype(np.float32)
    arr_bytes = (arr * 32767).astype(np.int16).tobytes()
    gated = audio_tools.gate_audio_bytes(arr_bytes, gate_probability=1.0, gate_length_ms=10, sample_rate=sr)
    assert isinstance(gated, bytes)
    assert len(gated) == len(arr_bytes)
    assert gated.count(b'\x00') > 0


def test_bit_crush_audio_bytes():
    arr = (np.random.rand(1000) * 2 - 1).astype(np.float32)
    arr_bytes = (arr * 32767).astype(np.int16).tobytes()
    crushed = audio_tools.bit_crush_audio_bytes(arr_bytes, intensity=1.0)
    assert isinstance(crushed, bytes)
    assert len(crushed) == len(arr_bytes)


def test_apply_speech_hallucination_processing():
    sr = 8000
    arr = (np.random.rand(sr) * 2 - 1).astype(np.float32)
    arr_bytes = (arr * 32767).astype(np.int16).tobytes()
    hallucinated = audio_tools.apply_speech_hallucination_processing(arr_bytes, hallucination_intensity=0.5, sample_rate=sr)
    assert isinstance(hallucinated, bytes)
    assert len(hallucinated) > 0


def test_save_wav_from_bytes(tmp_path):
    sr = 8000
    arr = (np.random.rand(sr) * 2 - 1).astype(np.float32)
    arr_bytes = (arr * 32767).astype(np.int16).tobytes()
    out_path = tmp_path / "test.wav"
    audio_tools.save_wav_from_bytes(str(out_path), arr_bytes, sample_rate=sr)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_find_wav_files(tmp_path):
    f1 = tmp_path / "a.wav"
    f2 = tmp_path / "b.WAV"
    f3 = tmp_path / "c.mp3"
    f1.write_bytes(b"data")
    f2.write_bytes(b"data")
    f3.write_bytes(b"data")
    files = audio_tools.find_wav_files(str(tmp_path), None)
    assert str(f1) in files
    assert str(f2) in files
    assert str(f3) not in files


def test_extract_non_silent_segments():
    sr = 8000
    audio = np.zeros(sr)
    audio[100:200] = 1.0
    segments = audio_tools.extract_non_silent_segments(audio, sr, min_duration=0.005)
    assert isinstance(segments, list)
    assert any(len(seg) >= int(0.005 * sr) for seg in segments)


def test_create_audio_mosaic_chain_artifact(tmp_path, monkeypatch):
    # Mock AudioChainArtifactFile and file I/O
    class DummyArtifact:
        def __init__(self, path, sam_r):
            self._path = path
            self.sample_rate = sam_r
            self.rainbow_color = the_rainbow_table_colors['Z']
            self.chain_artifact_file_type = ChainArtifactFileType.AUDIO
        def get_artifact_path(self):
            return self._path
    sr = 8000
    arr = np.random.rand(sr * 2).astype(np.float32)
    paths = []
    for i in range(3):
        p = tmp_path / f"test_{i}.wav"
        sf.write(str(p), arr, sr)
        paths.append(str(p))
    artifacts = [DummyArtifact(p, sr) for p in paths]
    monkeypatch.setenv("AGENT_WORK_PRODUCT_BASE_PATH", str(tmp_path))
    artifact = audio_tools.create_audio_mosaic_chain_artifact(artifacts, 50, 1.0, thread_id="tst")
    assert isinstance(artifact, AudioChainArtifactFile)
    assert os.path.exists(artifact.get_artifact_path())
    assert artifact.duration > 0



def test_create_blended_audio_chain_artifact(tmp_path, monkeypatch):
    # Prepare dummy mosaic artifact
    class DummyArtifact(AudioChainArtifactFile):
        def get_artifact_path(self, with_file_name=True):
            return str(tmp_path / "mosaic.wav")
    sr = 8000
    arr = np.random.rand(sr).astype(np.float32)
    sf.write(tmp_path / "mosaic.wav", arr, sr)
    monkeypatch.setenv("AGENT_WORK_PRODUCT_BASE_PATH", str(tmp_path))
    mosaic = DummyArtifact(
        base_path=str(tmp_path),
        rainbow_color=the_rainbow_table_colors['Z'],
        chain_artifact_file_type=ChainArtifactFileType.AUDIO,
        file_name="mosaic.wav",
        artifact_path=str(tmp_path / "mosaic.wav"),
        duration=1.0,
        sample_rate=sr,
        channels=1,
        bit_depth=16
    )
    artifact = audio_tools.create_blended_audio_chain_artifact(mosaic, 0.5, thread_id="tst")
    assert isinstance(artifact, AudioChainArtifactFile)


def test_generate_noise_returns_bytes_for_types():
    for ntype in (NoiseType.WHITE, NoiseType.PINK, NoiseType.BROWN):
        b = audio_tools.generate_noise(0.01, ntype, mix_level=0.1, sample_rate=44100)
        assert isinstance(b, (bytes, bytearray))
        assert len(b) > 0


def test_bit_crush_reduces_dynamic_range():
    b = make_sine_bytes(0.02)
    out = audio_tools.bit_crush_audio_bytes(b, intensity=0.8)
    assert isinstance(out, (bytes, bytearray))
    assert len(out) == len(b)
    # ensure output is different from input (not a strict guarantee, but highly likely)
    assert out != b


def test_micro_stutter_inserts_repeated_chunk(monkeypatch):
    b = make_sine_bytes(0.2)
    # force stutter every window and pick deterministic start
    monkeypatch.setattr(np.random, 'random', lambda: 0.0)
    monkeypatch.setattr(np.random, 'randint', lambda a, b: 0)
    out = audio_tools.micro_stutter_audio_bytes(b, stutter_probability=1.0, stutter_length_ms=10, sample_rate=44100)
    assert isinstance(out, (bytes, bytearray))
    # output should be longer than input because of inserted stutter
    assert len(out) >= len(b)


def test_gate_audio_bytes_raw_pcm_zeroes_region(tmp_path):
    sr = 1000
    duration = 1.0
    samples = int(sr * duration)
    arr = np.ones(samples, dtype=np.int16) * 1000
    b = arr.tobytes()
    # gate from 0.2 to 0.4 sec
    out = audio_tools.gate_audio_bytes(b, start_sec=0.2, end_sec=0.4, sample_rate=sr)
    out_arr = np.frombuffer(out, dtype=np.int16)
    start_idx = int(round(0.2 * sr))
    end_idx = int(round(0.4 * sr))
    # region should be zeroed
    assert np.all(out_arr[start_idx:end_idx] == 0)


def test_gate_audio_bytes_wav_container_zeroes_region(tmp_path):
    sr = 44100
    duration = 0.1
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 440 * t)
    buf = io.BytesIO()
    sf.write(buf, sig, sr, format='WAV')
    wav_bytes = buf.getvalue()
    out = audio_tools.gate_audio_bytes(wav_bytes, start_sec=0.01, end_sec=0.02)
    # read back and ensure section is zero
    buf2 = io.BytesIO(out)
    data, file_sr = sf.read(buf2, dtype='float32')
    start_idx = int(round(0.01 * file_sr))
    end_idx = int(round(0.02 * file_sr))
    assert np.allclose(data[start_idx:end_idx], 0.0, atol=1e-3)


def test_find_wav_files_and_prefix(tmp_path):
    d = tmp_path / "adir"
    d.mkdir()
    f1 = d / "a.wav"
    f2 = d / "prefix_b.wav"
    (d / "other.txt").write_text("x")
    sf.write(str(f1), np.zeros(10), 44100)
    sf.write(str(f2), np.zeros(10), 44100)
    all_files = audio_tools.find_wav_files(str(d), None)
    assert any(str(f1) in p for p in all_files)
    pref_files = audio_tools.find_wav_files(str(d), "prefix_")
    assert any("prefix_b.wav" in p for p in pref_files)


def test_extract_non_silent_segments_monkeypatched(monkeypatch):
    # create dummy audio and have librosa.effects.split return two intervals
    audio = np.arange(1000).astype(np.float32)
    monkeypatch.setattr(audio_tools.librosa.effects, 'split', lambda a, top_db: np.array([[0, 100], [200, 500]]))
    segs = audio_tools.extract_non_silent_segments(audio, sr=1000, min_duration=0.05, top_db=30)
    # min_duration 0.05*1000=50 samples, both intervals satisfy
    assert len(segs) == 2
    assert np.array_equal(segs[0], audio[0:100])


def test_create_audio_mosaic_chain_artifact_error_cases(tmp_path, monkeypatch):
    # empty segments list -> ValueError
    with pytest.raises(ValueError):
        audio_tools.create_audio_mosaic_chain_artifact([], 100, 1.0)

    # segments provided but files missing -> FileNotFoundError
    class DummySeg:
        def __init__(self):
            self.sample_rate = 44100
            self.file_name = "noexist.wav"
            self.rainbow_color = None
            self.chain_artifact_file_type = type('X', (), {'value': 'wav'})
            self.get_artifact_path = lambda: str(tmp_path / "noexist.wav")
    seg = DummySeg()
    with pytest.raises(FileNotFoundError):
        audio_tools.create_audio_mosaic_chain_artifact([seg], 100, 0.1)


def test_apply_speech_hallucination_processing_monkeypatched(monkeypatch):
    # monkeypatch heavy functions to be pass-throughs
    # ensure mocked generate_noise accepts sample_rate as optional kwarg
    monkeypatch.setattr(audio_tools, 'generate_noise', lambda duration_seconds, noise_type, mix_level, sample_rate=44100, freq_low=300, freq_high=3400: make_sine_bytes(0.02, sample_rate))
    monkeypatch.setattr(audio_tools, 'pitch_shift_audio_bytes', lambda audio, cents, sample_rate: audio)
    monkeypatch.setattr(audio_tools, 'micro_stutter_audio_bytes', lambda audio, stutter_probability, stutter_length_ms=50, sample_rate=44100: audio)
    monkeypatch.setattr(audio_tools, 'gate_audio_bytes', lambda audio, start_sec=None, end_sec=None, sample_rate=None, gate_probability=None, gate_length_ms=100: audio)
    monkeypatch.setattr(audio_tools, 'bit_crush_audio_bytes', lambda audio, intensity=0.5: audio)
    b = make_sine_bytes(0.05)
    out = audio_tools.apply_speech_hallucination_processing(b, hallucination_intensity=0.5, sample_rate=44100)
    assert isinstance(out, (bytes, bytearray))


def test_select_random_segment_audio_no_files(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_tools, 'find_wav_files', lambda root, prefix: [])
    # Should simply return without raising and create no output dir
    out_dir = tmp_path / "outseg"
    audio_tools.select_random_segment_audio(str(tmp_path), min_duration=0.01, num_segments=2, output_dir=str(out_dir))
    assert not out_dir.exists()


def test_select_random_segment_audio_with_files(tmp_path, monkeypatch):
    # Create two fake wav files
    sr = 8000
    arr = np.random.rand(sr).astype(np.float32)
    f1 = tmp_path / "a.wav"
    f2 = tmp_path / "b.wav"
    sf.write(str(f1), arr, sr)
    sf.write(str(f2), arr, sr)
    monkeypatch.setattr(audio_tools, 'find_wav_files', lambda root, prefix: [str(f1), str(f2)])
    # Monkeypatch extract_non_silent_segments to return one segment per file
    def fake_extract(audio, sr_in, min_duration, top_db=30):
        return [audio[0:100]]
    monkeypatch.setattr(audio_tools, 'extract_non_silent_segments', fake_extract)
    # Run selection
    out_dir = tmp_path / "segments_out"
    audio_tools.select_random_segment_audio(str(tmp_path), min_duration=0.001, num_segments=2, output_dir=str(out_dir))
    # Output dir should exist (may have 0 or more files depending on logic), but ensure it was created
    assert out_dir.exists()


def test_get_audio_segments_as_chain_artifacts(tmp_path, monkeypatch):
    # Monkeypatch find_wav_files to return two paths
    p1 = tmp_path / "one.wav"
    p2 = tmp_path / "two.wav"
    sr = 8000
    arr = np.random.rand(sr * 2).astype(np.float32)
    sf.write(str(p1), arr, sr)
    sf.write(str(p2), arr, sr)
    monkeypatch.setenv('MANIFEST_PATH', str(tmp_path))
    monkeypatch.setattr(audio_tools, 'find_wav_files', lambda root, prefix: [str(p1), str(p2)])
    # Monkeypatch librosa.load to return audio and sample rate
    monkeypatch.setattr(audio_tools.librosa, 'load', lambda path, sr=None: (np.random.rand(1000).astype(np.float32), sr or 44100))
    # Monkeypatch extract_non_silent_segments to return a couple segments
    monkeypatch.setattr(audio_tools, 'extract_non_silent_segments', lambda audio, sr, min_duration: [audio[0:100], audio[200:400]])
    # Monkeypatch sf.write to actually write files (it already does) but ensure output dir env is set
    monkeypatch.setenv('AGENT_WORK_PRODUCT_BASE_PATH', str(tmp_path))
    color = the_rainbow_table_colors['R']
    artifacts = audio_tools.get_audio_segments_as_chain_artifacts(0.01, 2, color, thread_id="tst")
    assert isinstance(artifacts, list)
    # either zero or more artifacts, but function should return list (not raise)


def test_create_random_audio_mosaic_no_wavs(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_tools, 'find_wav_files', lambda root, prefix: [])
    out = tmp_path / "mosaic.wav"
    audio_tools.create_random_audio_mosaic(str(tmp_path), slice_duration_ms=50, target_length_sec=0.5, output_path=str(out))
    assert not out.exists()


def test_select_random_segment_audio_writes(tmp_path, monkeypatch):
    # Create 1 wav file with longer audio and make extract_non_silent_segments return several segments
    sr = 8000
    arr = np.random.rand(sr * 2).astype(np.float32)
    wav = tmp_path / "long.wav"
    sf.write(str(wav), arr, sr)

    monkeypatch.setattr(audio_tools, 'find_wav_files', lambda root, prefix: [str(wav)])
    # librosa.load returns the array
    monkeypatch.setattr(audio_tools.librosa, 'load', lambda path, sr=None: (arr, sr or 44100))
    # extract_non_silent_segments returns multiple small segments
    monkeypatch.setattr(audio_tools, 'extract_non_silent_segments', lambda audio, sr, min_duration: [audio[0:100], audio[200:300], audio[400:500]])
    out_dir = tmp_path / "segout"
    audio_tools.select_random_segment_audio(str(tmp_path), min_duration=0.001, num_segments=2, output_dir=str(out_dir))
    assert out_dir.exists()
    # There should be at least one file written
    written = list(out_dir.glob('*.wav'))
    assert len(written) >= 1


def test_select_random_segment_audio_handles_load_error(tmp_path, monkeypatch):
    # Create a wav file but make librosa.load raise to hit exception branch
    sr = 8000
    arr = np.random.rand(sr).astype(np.float32)
    wav = tmp_path / "bad.wav"
    sf.write(str(wav), arr, sr)
    monkeypatch.setattr(audio_tools, 'find_wav_files', lambda root, prefix: [str(wav)])
    def fake_load(path, sr=None):
        raise RuntimeError("load failed")
    monkeypatch.setattr(audio_tools.librosa, 'load', fake_load)
    # Should not raise
    audio_tools.select_random_segment_audio(str(tmp_path), min_duration=0.001, num_segments=1, output_dir=str(tmp_path / 'o'))


def test_get_audio_segments_as_chain_artifacts_returns_empty(monkeypatch, tmp_path):
    # No wav files -> empty
    monkeypatch.setenv('MANIFEST_PATH', str(tmp_path))
    monkeypatch.setattr(audio_tools, 'find_wav_files', lambda root, prefix: [])
    color = the_rainbow_table_colors['R']
    res = audio_tools.get_audio_segments_as_chain_artifacts(0.1, 1, color)
    assert res == []


def test_create_random_audio_mosaic_success(tmp_path, monkeypatch):
    # Create several wav files with enough length
    sr = 44100
    arr = np.random.rand(sr * 2).astype(np.float32)
    files = []
    for i in range(5):
        p = tmp_path / f"f{i}.wav"
        sf.write(str(p), arr, sr)
        files.append(str(p))
    monkeypatch.setattr(audio_tools, 'find_wav_files', lambda root, prefix: files)
    out = tmp_path / "mosaic_out.wav"
    audio_tools.create_random_audio_mosaic(str(tmp_path), slice_duration_ms=100, target_length_sec=0.5, output_path=str(out))
    # mosaic should be created
    assert out.exists() is True


def test_gate_audio_bytes_no_gate_due_to_probability(monkeypatch):
    # raw PCM case: if random.random() >= gate_probability we should get original bytes
    sr = 1000
    samples = sr
    arr = np.ones(samples, dtype=np.int16)
    b = arr.tobytes()
    monkeypatch.setattr('random.random', lambda: 0.99)
    out = audio_tools.gate_audio_bytes(b, gate_probability=0.5, gate_length_ms=10, sample_rate=sr)
    assert out == b


def test_gate_audio_bytes_multichannel_wav(tmp_path):
    # create stereo audio and ensure gating zeroes both channels
    sr = 1000
    t = np.linspace(0, 0.1, int(0.1 * sr), endpoint=False)
    sig = np.column_stack([0.5 * np.sin(2 * np.pi * 440 * t), 0.25 * np.sin(2 * np.pi * 220 * t)])
    buf = io.BytesIO()
    sf.write(buf, sig, sr, format='WAV')
    wav_bytes = buf.getvalue()
    out = audio_tools.gate_audio_bytes(wav_bytes, start_sec=0.02, end_sec=0.04)
    data, file_sr = sf.read(io.BytesIO(out), dtype='float32')
    sidx = int(round(0.02 * file_sr))
    eidx = int(round(0.04 * file_sr))
    # both channels should be zeroed in that window
    assert data.ndim == 2
    assert np.allclose(data[sidx:eidx, 0], 0.0, atol=1e-3)
    assert np.allclose(data[sidx:eidx, 1], 0.0, atol=1e-3)


def test_safe_normalize_and_clip_behaviour():
    fn = audio_tools._safe_normalize_and_clip
    # empty array
    empty = np.array([], dtype=float)
    assert fn(empty).size == 0
    # zeros -> zeros
    z = np.zeros(10)
    assert np.all(fn(z) == 0)
    # with NaN/inf
    arr = np.array([np.nan, np.inf, -np.inf, 1.0, -2.0])
    out = fn(arr)
    assert not np.isnan(out).any()
    assert np.isfinite(out).all()
    # normalized to max abs 1.0
    assert np.max(np.abs(out)) <= 1.0


def test_select_random_segment_audio_sf_write_exception(tmp_path, monkeypatch):
    # ensure that if sf.write raises, select_random_segment_audio handles it
    sr = 8000
    arr = np.random.rand(sr).astype(np.float32)
    wav = tmp_path / "file.wav"
    sf.write(str(wav), arr, sr)
    monkeypatch.setattr(audio_tools, 'find_wav_files', lambda root, prefix: [str(wav)])
    monkeypatch.setattr(audio_tools.librosa, 'load', lambda path, sr=None: (arr, sr or 44100))
    monkeypatch.setattr(audio_tools, 'extract_non_silent_segments', lambda audio, sr, min_duration: [audio[0:100]])
    # make sf.write raise for the segment write
    def fake_write(path, data, sr_in, subtype=None):
        if 'segment_' in str(path):
            raise RuntimeError("write failed")
        return None
    monkeypatch.setattr(audio_tools.sf, 'write', fake_write)
    out_dir = tmp_path / "outseg2"
    # Should not raise
    audio_tools.select_random_segment_audio(str(tmp_path), min_duration=0.001, num_segments=1, output_dir=str(out_dir))
    # Directory may exist even if write failed
    assert out_dir.exists()
