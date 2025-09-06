import numpy as np

from pydantic import BaseModel

class RainbowAudioFeature(BaseModel):

    audio_file_path: str
    segment_start_time: float
    segment_end_time: float
    duration: float
    duration_samples: int | None = None
    silence_analysis: dict | None = None
    is_mostly_silence: bool | None = None
    non_silence_ratio: float | None = None
    silence_gaps: list[float] | None = None
    non_silence_regions: list[tuple[float, float]] | None = None
    peak_amplitude: float | None = None
    rms_energy: float | None = None
    spectral_centroid: float | None = None
    zero_crossing_rate: float | None = None
    tempo: float | None = None
    mfcc: np.ndarray | None = None # type: ignore
    chroma: np.ndarray | None = None # type: ignore
    spectral_contrast: np.ndarray | None = None # type: ignore
    onset_frames: np.ndarray | None = None # type: ignore
    onset_strength: np.ndarray | None = None # type: ignore
    harmonic_ratio: float | None = None
    attack_time: float | None = None
    decay_profile: np.ndarray | None = None # type: ignore
    silence_confidence: float | None = None

    def __init__(self, **data):
        super().__init__(**data)

    model_config = {
        "arbitrary_types_allowed": True
    }
