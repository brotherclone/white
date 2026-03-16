## 1. Grain Synthesizer Core
- [x] 1.1 Write `training/tools/grain_synthesizer.py`:
      - `load_grain_pool(color, top_n, clap_parquet, meta_parquet) -> list[dict]`:
        calls `retrieve_by_color()` from `retrieve_samples.py`; returns result dicts
        with `source_audio_file`, `start_seconds`, `end_seconds`, `match`
      - `extract_grain(source_path, segment_start, segment_end, grain_dur=1.0) -> np.ndarray`:
        loads via `soundfile.read()`, picks a random offset within [segment_start,
        segment_end - grain_dur], returns float32 array
      - `hann_crossfade(grains, sr, crossfade_ms=50) -> np.ndarray`:
        applies Hann-windowed crossfades between adjacent grains; concatenates into
        a single array
      - `synthesize(color, duration_s, top_n, output_path, clap_parquet, meta_parquet,
        seed, grain_dur_s)`: end-to-end — load pool, select grains randomly from pool
        with replacement until `duration_s` reached, crossfade, write WAV + grain_map.yml
- [x] 1.2 `grain_map.yml` schema:
      ```yaml
      color: Red
      duration_s: 30.0
      grain_dur_s: 1.0
      seed: 42
      grains:
        - rank: 1
          segment_id: red_seg_003
          source: /path/to/source.wav
          offset_s: 4.72
          match: 0.8841
      ```
- [x] 1.3 Handle mono/stereo consistently: if pool contains mixed mono/stereo, convert
      all grains to stereo before crossfade

## 2. CLI
- [x] 2.1 Add `argparse` CLI:
      - `--color` (required)
      - `--duration` (default 30, seconds)
      - `--top-n` (default 20, segments in grain pool)
      - `--output` (default `./grain_output/<color>_texture.wav`)
      - `--seed` (default 42)
      - `--grain-dur` (default 1.0, seconds)
      - `--parquet` (override CLAP parquet path)
      - `--crossfade-ms` (default 50)
- [x] 2.2 Print grain pool summary and progress during synthesis

## 3. Tests
- [x] 3.1 `test_extract_grain`: given a synthetic WAV of known length, extract_grain
      returns an array of the correct duration and stays within segment bounds
- [x] 3.2 `test_hann_crossfade_length`: output length matches expected duration given
      grain count, grain duration, and crossfade length
- [x] 3.3 `test_hann_crossfade_no_clicks`: RMS of crossfade boundaries does not exceed
      RMS of adjacent grains (no energy spike at join points)
- [x] 3.4 `test_synthesize_writes_wav_and_map`: end-to-end with stub pool (no parquet
      needed), writes WAV and grain_map.yml; WAV duration is within 1 grain of target
- [x] 3.5 `test_mono_stereo_normalization`: pool with mixed mono/stereo produces stereo
      output without error
- [x] 3.6 `test_empty_pool_raises`: if retrieve_by_color returns no results, synthesize
      raises a clear error before attempting audio operations
