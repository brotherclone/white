"""Small audio I/O helpers that use soundfile (libsndfile) under the hood.

This avoids `audioread` as a backend and thus prevents `aifc`/`sunau` deprecation warnings.
Provides a `load_audio` function with a minimal compatible subset of `librosa.load` behavior.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

try:
    import librosa
except EnvironmentError:
    librosa = None


def load_audio(path_or_file, sr: int | None = None, mono: bool = True):
    """Load audio using soundfile. Returns (y, sr).

    - path_or_file: filename or file-like object supported by soundfile.
    - sr: target sample rate. If provided and different, data will be resampled.
    - mono: if True, convert multi-channel audio to mono by averaging channels.

    The returned `y` is a numpy float32 array with samples in -1.0..1.0 range,
    similar to librosa.load.
    """
    data, file_sr = sf.read(path_or_file, dtype="float32", always_2d=False)
    data = np.asarray(data, dtype=np.float32)
    if mono and data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr is not None and file_sr != sr:
        if librosa is not None:
            data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
            file_sr = sr
        else:
            # As a fallback, use numpy-based resampling (less optimal)
            # simple linear interpolation
            duration = data.shape[0] / file_sr
            target_samples = int(round(duration * sr))
            if target_samples <= 0:
                data = np.array([], dtype=np.float32)
                file_sr = sr
            else:
                old_indices = np.linspace(0.0, 1.0, num=data.shape[0])
                new_indices = np.linspace(0.0, 1.0, num=target_samples)
                data = np.interp(new_indices, old_indices, data).astype(np.float32)
                file_sr = sr
    return data, file_sr
