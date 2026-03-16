#!/usr/bin/env python3
"""
Spike: Stable Audio Open — Chromatic Text Prompt Evaluation

Generates 3 clips × 8 colors = 24 clips using StableAudioPipeline (diffusers),
scores each with CLAP + Refractor in audio-only mode, and writes results to
training/spikes/stable-audio-prompts/spike_results.jsonl.

Model: stabilityai/stable-audio-open-1.0
License: Stability AI Community License (commercial use requires separate license
         if annual revenue > $1M; see https://stability.ai/community-license-agreement)

Usage:
    # Dry run — generate 1 clip for Red only (fast, cheap)
    modal run training/modal_stable_audio_spike.py --dry-run

    # Full spike — 24 clips across 8 colors
    modal run training/modal_stable_audio_spike.py

    # Download results after run
    modal volume get white-training-data stable-audio-spike/ ./spike_output/
"""

import json
import modal

app = modal.App("white-stable-audio-spike")

volume = modal.Volume.from_name("white-training-data", create_if_missing=True)

CACHE_DIR = "/cache"
SPIKE_DIR = f"{CACHE_DIR}/stable-audio-spike"

spike_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install(
        "torch==2.4.0",
        "diffusers==0.30.3",
        "transformers==4.46.3",
        "accelerate",
        "safetensors",
        "soundfile",
        "numpy",
        "huggingface_hub",
        # (transformers already pulled in above — ClapModel uses laion/larger_clap_music)
        # Stable Audio scheduler dependency
        "torchsde",
        # Refractor ONNX scoring
        "onnxruntime",
    )
)

# ---------------------------------------------------------------------------
# Chromatic prompts
# ---------------------------------------------------------------------------

CHROMATIC_PROMPTS = {
    "Red": (
        "sparse ambient piano, melancholic, slow, introspective, past memories, "
        "wistful, fading, quiet reverb, nostalgic"
    ),
    "Orange": (
        "warm nostalgic folk texture, familiar place, rootsy acoustic guitar, "
        "gentle, grounded, orange sunset, belonging"
    ),
    "Yellow": (
        "present moment, bright acoustic texture, open fields, optimistic, "
        "forward motion, daylight, organic, alive"
    ),
    "Green": (
        "organic nature sounds, present-tense growth, living breathing texture, "
        "forest ambience, gentle movement, verdant"
    ),
    "Blue": (
        "present urban ambient, cool minimal electronic, observational, "
        "street sounds dissolved into drone, detached, blue hour"
    ),
    "Indigo": (
        "alien drone, suspended time, no tonal anchor, unreal space, "
        "dissonant shimmer, liminal void, indigo darkness"
    ),
    "Violet": (
        "dreaming, liminal between-states, soft electronic wash, future imagined, "
        "violet dusk, floating, aspirational, dissolving"
    ),
    "Black": (
        "dark ambient void, formless, subterranean rumble, undefined, no melody, "
        "deep silence with texture, black space"
    ),
}

CLIPS_PER_COLOR = 3
AUDIO_DURATION_S = 30.0
SAMPLE_RATE = 44100

# ---------------------------------------------------------------------------
# Generation function (runs on GPU)
# ---------------------------------------------------------------------------


@app.function(
    image=spike_image,
    gpu="A10G",
    timeout=3600,
    volumes={CACHE_DIR: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate_and_score(
    color: str,
    prompt: str,
    clip_index: int,
    seed: int,
    dry_run: bool = False,
) -> dict:
    """Generate one audio clip and score it with CLAP + Refractor."""
    import time
    import os
    import numpy as np
    import soundfile as sf
    import torch
    from diffusers import StableAudioPipeline

    os.makedirs(SPIKE_DIR, exist_ok=True)

    print(f"[{color} clip {clip_index}] seed={seed}")
    print(f"  Prompt: {prompt[:80]}...")

    # Load pipeline (cached after first call in the container lifetime)
    print("  Loading StableAudioPipeline...")
    pipe = StableAudioPipeline.from_pretrained(
        "stabilityai/stable-audio-open-1.0",
        torch_dtype=torch.float16,
        cache_dir=f"{CACHE_DIR}/hf_cache",
    )
    pipe = pipe.to("cuda")

    # Generate
    t0 = time.time()
    generator = torch.Generator("cuda").manual_seed(seed)
    output = pipe(
        prompt=prompt,
        negative_prompt="low quality, noise, distortion, speech, vocals",
        num_inference_steps=100,
        audio_end_in_s=AUDIO_DURATION_S,
        generator=generator,
    )
    generation_time_s = round(time.time() - t0, 2)
    print(f"  Generated in {generation_time_s}s")

    # Convert to numpy float32 stereo (N, 2)
    audio_tensor = output.audios[0]  # shape (channels, samples)
    audio_np = audio_tensor.T.float().cpu().numpy()  # (samples, channels)
    if audio_np.ndim == 1:
        audio_np = np.stack([audio_np, audio_np], axis=1)

    # Save WAV
    wav_filename = f"{color.lower()}_clip{clip_index}_seed{seed}.wav"
    wav_path = f"{SPIKE_DIR}/{wav_filename}"
    sf.write(wav_path, audio_np, SAMPLE_RATE)
    print(f"  Saved: {wav_path}")

    if dry_run:
        volume.commit()
        return {
            "color": color,
            "clip_index": clip_index,
            "seed": seed,
            "prompt": prompt,
            "wav_file": wav_filename,
            "generation_time_s": generation_time_s,
            "clap_embedding": None,
            "chromatic_match": None,
            "temporal": None,
            "spatial": None,
            "ontological": None,
            "confidence": None,
            "dry_run": True,
        }

    # CLAP embedding — use same model as corpus extraction (laion/larger_clap_music)
    print("  Computing CLAP embedding...")
    from transformers import ClapModel, ClapProcessor

    clap_processor = ClapProcessor.from_pretrained(
        "laion/larger_clap_music",
        cache_dir=f"{CACHE_DIR}/hf_cache",
    )
    clap_model = ClapModel.from_pretrained(
        "laion/larger_clap_music",
        cache_dir=f"{CACHE_DIR}/hf_cache",
    ).to("cuda")
    clap_model.eval()

    # Processor expects 48kHz — resample from 44.1kHz
    target_sr = 48000
    ratio = target_sr / SAMPLE_RATE
    n_orig = len(audio_np)
    n_target = int(n_orig * ratio)
    indices = np.linspace(0, n_orig - 1, n_target)
    mono = audio_np.mean(axis=1)
    mono_resampled = np.interp(indices, np.arange(n_orig), mono).astype(np.float32)

    with torch.no_grad():
        inputs = clap_processor(
            audios=mono_resampled, sampling_rate=target_sr, return_tensors="pt"
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        audio_emb = clap_model.audio_model(**inputs)
        pooled = audio_emb.pooler_output  # (1, hidden_size)
        clap_emb_np = (
            clap_model.audio_projection(pooled).cpu().float().numpy()[0]
        )  # (512,)

    # Refractor scoring (audio-only, null concept)
    print("  Scoring with Refractor...")

    # Refractor ONNX — download from HF dataset if not cached in volume
    onnx_path = f"{CACHE_DIR}/data/models/fusion_model.onnx"
    if not os.path.exists(onnx_path):
        from huggingface_hub import hf_hub_download

        onnx_path = hf_hub_download(
            repo_id="earthlyframes/white-training-data",
            filename="data/models/fusion_model.onnx",
            repo_type="dataset",
            local_dir=CACHE_DIR,
        )

    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path)
    null_concept = np.zeros(768, dtype=np.float32)
    null_midi = np.zeros(512, dtype=np.float32)
    null_lyric = np.zeros(768, dtype=np.float32)
    null_sounds_like = np.zeros(768, dtype=np.float32)

    inputs = {
        "concept_emb": null_concept[np.newaxis],
        "audio_emb": clap_emb_np[np.newaxis],
        "midi_emb": null_midi[np.newaxis],
        "lyric_emb": null_lyric[np.newaxis],
        "sounds_like_emb": null_sounds_like[np.newaxis],
        "has_concept": np.array([[0.0]], dtype=np.float32),
        "has_audio": np.array([[1.0]], dtype=np.float32),
        "has_midi": np.array([[0.0]], dtype=np.float32),
        "has_lyric": np.array([[0.0]], dtype=np.float32),
        "has_sounds_like": np.array([[0.0]], dtype=np.float32),
    }
    outputs = sess.run(None, inputs)

    # outputs: [temporal_dist, spatial_dist, ontological_dist] each shape (1, n_classes)
    def _softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    temporal_dist = _softmax(outputs[0][0])
    spatial_dist = _softmax(outputs[1][0])
    ontological_dist = _softmax(outputs[2][0])

    # Compute chromatic match against target (inlined from chord_pipeline.py)
    _CHROMATIC_TARGETS = {
        "Red": {
            "temporal": [0.8, 0.1, 0.1],
            "spatial": [0.8, 0.1, 0.1],
            "ontological": [0.1, 0.1, 0.8],
        },
        "Orange": {
            "temporal": [0.8, 0.1, 0.1],
            "spatial": [0.1, 0.8, 0.1],
            "ontological": [0.8, 0.1, 0.1],
        },
        "Yellow": {
            "temporal": [0.1, 0.8, 0.1],
            "spatial": [0.1, 0.8, 0.1],
            "ontological": [0.1, 0.1, 0.8],
        },
        "Green": {
            "temporal": [0.1, 0.8, 0.1],
            "spatial": [0.1, 0.8, 0.1],
            "ontological": [0.1, 0.8, 0.1],
        },
        "Blue": {
            "temporal": [0.8, 0.1, 0.1],
            "spatial": [0.1, 0.1, 0.8],
            "ontological": [0.1, 0.1, 0.8],
        },
        "Indigo": {
            "temporal": [0.1, 0.1, 0.8],
            "spatial": [0.8, 0.1, 0.1],
            "ontological": [0.1, 0.8, 0.1],
        },
        "Violet": {
            "temporal": [0.1, 0.8, 0.1],
            "spatial": [0.1, 0.1, 0.8],
            "ontological": [0.8, 0.1, 0.1],
        },
        "Black": {
            "temporal": [1 / 3, 1 / 3, 1 / 3],
            "spatial": [1 / 3, 1 / 3, 1 / 3],
            "ontological": [1 / 3, 1 / 3, 1 / 3],
        },
    }
    _TEMPORAL_MODES = ["past", "present", "future"]
    _SPATIAL_MODES = ["thing", "place", "person"]
    _ONTOLOGICAL_MODES = ["imagined", "forgotten", "known"]

    target = _CHROMATIC_TARGETS.get(color, _CHROMATIC_TARGETS["Black"])
    score_dict = {
        "temporal": {m: float(v) for m, v in zip(_TEMPORAL_MODES, temporal_dist)},
        "spatial": {m: float(v) for m, v in zip(_SPATIAL_MODES, spatial_dist)},
        "ontological": {
            m: float(v) for m, v in zip(_ONTOLOGICAL_MODES, ontological_dist)
        },
        "confidence": 1.0,
    }

    match = 0.0
    for dim, modes in [
        ("temporal", _TEMPORAL_MODES),
        ("spatial", _SPATIAL_MODES),
        ("ontological", _ONTOLOGICAL_MODES),
    ]:
        target_arr = np.array(target[dim])
        pred_arr = np.array([score_dict[dim][m] for m in modes])
        match += np.dot(pred_arr, target_arr)
    chromatic_match = float((match / 3.0) * (0.5 + 0.5 * score_dict["confidence"]))

    result = {
        "color": color,
        "clip_index": clip_index,
        "seed": seed,
        "prompt": prompt,
        "wav_file": wav_filename,
        "generation_time_s": generation_time_s,
        "chromatic_match": round(chromatic_match, 4),
        "temporal": [round(float(v), 4) for v in temporal_dist],
        "spatial": [round(float(v), 4) for v in spatial_dist],
        "ontological": [round(float(v), 4) for v in ontological_dist],
    }

    print(f"  chromatic_match={chromatic_match:.4f}")

    # Append to results JSONL in volume
    results_path = f"{SPIKE_DIR}/spike_results.jsonl"
    with open(results_path, "a") as f:
        f.write(json.dumps(result) + "\n")

    volume.commit()
    return result


# ---------------------------------------------------------------------------
# Orchestrator (local entrypoint)
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(dry_run: bool = False):
    """Run the full spike: 3 clips × 8 colors = 24 clips."""
    import random

    colors = list(CHROMATIC_PROMPTS.keys())
    if dry_run:
        colors = ["Red"]
        n_clips = 1
        print("DRY RUN — generating 1 clip (Red) only")
    else:
        n_clips = CLIPS_PER_COLOR
        print(
            f"FULL SPIKE — {len(colors)} colors × {n_clips} clips = {len(colors) * n_clips} total"
        )

    print(f"Duration: {AUDIO_DURATION_S}s per clip | GPU: A10G")
    print()

    # Build jobs
    rng = random.Random(42)
    jobs = []
    for color in colors:
        prompt = CHROMATIC_PROMPTS[color]
        for i in range(n_clips):
            seed = rng.randint(0, 2**31)
            jobs.append((color, prompt, i, seed))

    # Run in parallel
    results = list(
        generate_and_score.starmap(
            [(color, prompt, i, seed, dry_run) for color, prompt, i, seed in jobs]
        )
    )

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Color':<10} {'Clip':<5} {'Match':<8} {'Time(s)'}")
    print("-" * 35)
    for r in sorted(results, key=lambda x: (x["color"], x["clip_index"])):
        match_str = (
            f"{r['chromatic_match']:.4f}"
            if r.get("chromatic_match") is not None
            else "n/a"
        )
        print(
            f"{r['color']:<10} {r['clip_index']:<5} {match_str:<8} {r['generation_time_s']}"
        )

    if not dry_run:
        # Per-color summary
        from collections import defaultdict

        by_color = defaultdict(list)
        for r in results:
            if r.get("chromatic_match") is not None:
                by_color[r["color"]].append(r["chromatic_match"])

        print("\nPer-color summary (pass = chromatic_match > 0.4):")
        print(f"{'Color':<10} {'Mean':<8} {'Pass rate'}")
        print("-" * 30)
        for color in colors:
            scores = by_color.get(color, [])
            if scores:
                mean = sum(scores) / len(scores)
                pass_rate = sum(1 for s in scores if s > 0.4) / len(scores)
                print(
                    f"{color:<10} {mean:.4f}   {pass_rate:.0%} ({sum(1 for s in scores if s > 0.4)}/{len(scores)})"
                )

    print("\nResults written to Modal Volume: stable-audio-spike/spike_results.jsonl")
    print(
        "Download with: modal volume get white-training-data stable-audio-spike/ ./spike_output/"
    )
