## 1. Prompt Design
- [ ] 1.1 Write `CHROMATIC_PROMPTS` dict: one text prompt per color derived from the
      `CHROMATIC_TARGETS` temporal/spatial/ontological mode descriptions:
      ```python
      CHROMATIC_PROMPTS = {
          "Red":    "melancholic, past-tense, sparse ambient piano, slow, introspective",
          "Orange": "warm nostalgia, familiar memory, gentle folk texture, rootsy",
          "Yellow": "optimistic future, expansive, bright acoustic, forward motion",
          "Green":  "present-tense growth, organic, living texture, nature sounds",
          "Blue":   "present moment, urban, observational, cool electronic ambient",
          "Indigo": "unreal, alien, suspended time, drone, no tonal anchor",
          "Violet": "dreaming, liminal, between states, soft electronic wash",
          "Black":  "void, formless, dark ambient, undefined, subterranean",
      }
      ```
- [ ] 1.2 Get approval on prompt text before running GPU compute

## 2. Modal Spike Script
- [ ] 2.1 Write `training/modal_musicgen_spike.py`:
      - MusicGen-Medium (1.5B) via `audiocraft` on Modal A10G
      - Generate 3 clips × 8 colors = 24 clips at 30s each
      - Save as WAV to Modal Volume `white-training-data/musicgen-spike/`
      - Compute CLAP embedding for each output via CLAP model
      - Score each CLAP embedding with Refractor (audio-only mode, null concept)
      - Write results to `musicgen_results.jsonl`: color, prompt, clip_id,
        chromatic_match, temporal, spatial, ontological, generation_time_s
- [ ] 2.2 Run on Modal; download results JSONL

## 3. Analysis
- [ ] 3.1 Compute pass rate per color: % clips with chromatic_match > 0.4
- [ ] 3.2 Compute mean chromatic_match per color; compare to corpus baseline
      (mean score of top-20 retrieved segments per color from retrieve_by_color)
- [ ] 3.3 Note generation time and estimated GPU cost per clip

## 4. Spike Report
- [ ] 4.1 Write `training/spikes/musicgen-prompts/spike_report.md`:
      - Section 1: Prompts used per color
      - Section 2: Results table (color, pass_rate, mean_match, vs_corpus_baseline)
      - Section 3: License assessment (CC-BY-NC implications for Earthly Frames)
      - Section 4: Go / no-go recommendation with threshold rationale
      - Section 5: If go — sketch interface for `add-musicgen-chromatic-synthesis` proposal
