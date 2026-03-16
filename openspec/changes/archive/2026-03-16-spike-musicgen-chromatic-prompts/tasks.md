## 1. Prompt Design
- [x] 1.1 Write `CHROMATIC_PROMPTS` dict in `modal_stable_audio_spike.py`:
      one text prompt per color derived from CHROMATIC_TARGETS axes
- [x] 1.2 Prompts approved — baked into Modal script

## 2. Modal Spike Script
- [x] 2.1 Write `training/modal_stable_audio_spike.py`:
      - Stable Audio Open (`stabilityai/stable-audio-open-1.0`) via diffusers on Modal A10G
      - Generate 3 clips × 8 colors = 24 clips at 30s each (44.1kHz stereo)
      - Save WAVs to Modal Volume `white-training-data/stable-audio-spike/`
      - Compute CLAP embedding (laion-clap) for each output
      - Score with Refractor ONNX (audio-only, null concept)
      - Write results to `spike_results.jsonl`: color, prompt, clip_id,
        chromatic_match, temporal, spatial, ontological, generation_time_s
      - `--dry-run` flag: generate 1 Red clip only for validation
- [ ] 2.2 Run on Modal; download results JSONL:
      ```
      modal run training/modal_stable_audio_spike.py --dry-run  # validate
      modal run training/modal_stable_audio_spike.py            # full run
      modal volume get white-training-data stable-audio-spike/ ./spike_output/
      ```

## 3. Analysis
- [x] 3.1 Write `training/spikes/stable-audio-prompts/analyze_results.py`:
      - Pass rate per color (chromatic_match > 0.4)
      - Mean chromatic_match per color vs corpus baseline
      - Go/no-go verdict at ≥50% overall pass rate
- [ ] 3.2 Run analysis after downloading results:
      ```
      python training/spikes/stable-audio-prompts/analyze_results.py \
          --results ./spike_output/spike_results.jsonl --corpus-baseline
      ```

## 4. Spike Report
- [ ] 4.1 `training/spikes/stable-audio-prompts/spike_report.md` auto-generated
      by analyze_results.py after results downloaded; review and commit
