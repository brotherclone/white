#!/usr/bin/env python3
"""
Phase 3.1: CLAP Audio Embedding Extraction on Modal

Extracts CLAP audio embeddings (512-dim) from the media parquet on HuggingFace.
Streams row-group-by-row-group to avoid loading the full 15 GB into memory.

Uses a Modal Volume to cache the media parquet download between runs.

Usage:
    # Dry run (download + verify first row group, no GPU encoding)
    modal run training/modal_clap_extraction.py --dry-run

    # Full extraction
    modal run training/modal_clap_extraction.py

    # Start from a specific row group (resume after failure)
    modal run training/modal_clap_extraction.py --start-row-group 5
"""

import modal

app = modal.App("white-clap-embeddings")

volume = modal.Volume.from_name("white-training-data", create_if_missing=True)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1")  # needed by soundfile/librosa
    .pip_install(
        "torch",
        "transformers",
        "pandas",
        "pyarrow",
        "numpy",
        "tqdm",
        "safetensors",
        "huggingface_hub",
        "librosa",
        "soundfile",
    )
)

CLAP_MODEL = "laion/larger_clap_music"
HF_REPO = "earthlyframes/white-training-data"
HF_MEDIA_PATH = "data/training_segments_media.parquet"
CACHE_DIR = "/cache"
CLAP_DIM = 512


@app.function(
    image=gpu_image,
    gpu="A10G",
    timeout=7200,  # 2 hours max
    retries=0,
    volumes={CACHE_DIR: volume},
)
def extract_clap_embeddings(
    dry_run: bool = False,
    start_row_group: int = 0,
    batch_size: int = 4,
):
    """Extract CLAP audio embeddings, streaming row groups from media parquet."""
    import torch
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import librosa
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm

    # --- 1. Download media parquet (cached in volume) ---
    cached_path = Path(CACHE_DIR) / "training_segments_media.parquet"

    if cached_path.exists():
        size_gb = cached_path.stat().st_size / (1024**3)
        print(f"Using cached media parquet: {cached_path} ({size_gb:.1f} GB)")
    else:
        print(f"Downloading {HF_MEDIA_PATH} from {HF_REPO} (~15 GB)...")
        local_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_MEDIA_PATH,
            repo_type="dataset",
            local_dir=CACHE_DIR,
        )
        # hf_hub_download with local_dir puts it in a subdirectory
        downloaded = Path(local_path)
        if downloaded != cached_path:
            import shutil

            shutil.move(str(downloaded), str(cached_path))
        size_gb = cached_path.stat().st_size / (1024**3)
        print(f"Downloaded: {cached_path} ({size_gb:.1f} GB)")
        volume.commit()

    # --- 2. Inspect parquet structure ---
    pf = pq.ParquetFile(str(cached_path))
    num_row_groups = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows
    print(f"\nParquet: {total_rows} rows, {num_row_groups} row groups")

    # Columns we need (avoid loading the full 60 GB)
    needed_columns = ["segment_id", "audio_waveform", "audio_sample_rate", "has_audio"]

    if dry_run:
        print("\n--- DRY RUN: reading first row group ---")
        rg = pf.read_row_group(0, columns=needed_columns).to_pandas()
        print(f"Row group 0: {len(rg)} rows")
        has_audio = rg["has_audio"].sum()
        has_waveform = rg["audio_waveform"].notna().sum()
        print(f"  has_audio=True: {has_audio}")
        print(f"  audio_waveform non-null: {has_waveform}")
        if has_waveform > 0:
            sample = rg.loc[rg["audio_waveform"].notna()].iloc[0]
            audio = np.frombuffer(sample["audio_waveform"], dtype=np.float32)
            sr = sample["audio_sample_rate"]
            duration = len(audio) / sr
            print(f"  Sample: {len(audio)} samples, {sr} Hz, {duration:.1f}s")
        return {
            "status": "dry_run",
            "total_rows": total_rows,
            "row_groups": num_row_groups,
        }

    # --- 3. Load CLAP model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {mem_gb:.1f} GB")

    from transformers import ClapModel, ClapProcessor

    print("Loading CLAP model and processor...")
    processor = ClapProcessor.from_pretrained(CLAP_MODEL)
    model = ClapModel.from_pretrained(CLAP_MODEL)
    model = model.to(device)
    model.eval()
    print("CLAP loaded.")

    # --- 4. Process row groups ---
    all_segment_ids = []
    all_embeddings = []
    all_has_audio = []
    segments_processed = 0
    segments_with_audio = 0

    for rg_idx in range(start_row_group, num_row_groups):
        print(f"\n--- Row group {rg_idx}/{num_row_groups - 1} ---")
        rg_df = pf.read_row_group(rg_idx, columns=needed_columns).to_pandas()
        print(f"  Rows: {len(rg_df)}")

        for _, row in tqdm(rg_df.iterrows(), total=len(rg_df), desc=f"RG {rg_idx}"):
            seg_id = row["segment_id"]
            all_segment_ids.append(seg_id)

            audio_bytes = row["audio_waveform"]
            sample_rate = row["audio_sample_rate"]

            if audio_bytes is not None and row["has_audio"]:
                try:
                    waveform = np.frombuffer(audio_bytes, dtype=np.float32)

                    # Resample 44.1kHz → 48kHz (CLAP requirement)
                    sr = int(sample_rate)
                    if sr != 48000:
                        waveform = librosa.resample(
                            waveform, orig_sr=sr, target_sr=48000
                        )

                    inputs = processor(
                        audio=waveform,
                        sampling_rate=48000,
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    with torch.no_grad():
                        input_features = inputs["input_features"].to(device)
                        # Run audio encoder directly, then project + normalize
                        audio_out = model.audio_model(input_features=input_features)
                        pooled = audio_out.pooler_output  # [1, 768]
                        projected = model.audio_projection(pooled)  # [1, 512]
                        emb = torch.nn.functional.normalize(projected, dim=-1)
                        emb = emb.cpu().numpy()[0]  # [512]

                    all_embeddings.append(emb)
                    all_has_audio.append(True)
                    segments_with_audio += 1

                    if segments_with_audio == 1:
                        print(
                            f"  First embedding: shape={emb.shape}, norm={np.linalg.norm(emb):.4f}"
                        )
                        print(f"  Sample values: {emb[:5]}")
                except Exception as e:
                    print(f"  WARNING: Failed {seg_id}: {e}")
                    all_embeddings.append(np.zeros(CLAP_DIM, dtype=np.float32))
                    all_has_audio.append(False)
            else:
                all_embeddings.append(np.zeros(CLAP_DIM, dtype=np.float32))
                all_has_audio.append(False)

            segments_processed += 1

        # Free row group memory
        del rg_df

        print(
            f"  Progress: {segments_processed}/{total_rows} total, {segments_with_audio} with audio"
        )

        # Flush CUDA cache between row groups
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- 5. Build output ---
    print(
        f"\nBuilding output: {len(all_segment_ids)} segments, {segments_with_audio} with embeddings"
    )

    out_df = pd.DataFrame(
        {
            "segment_id": all_segment_ids,
            "audio_embedding": all_embeddings,
            "has_audio_embedding": all_has_audio,
        }
    )

    import io

    buf = io.BytesIO()
    out_df.to_parquet(buf)
    parquet_bytes = buf.getvalue()
    size_mb = len(parquet_bytes) / (1024 * 1024)
    print(f"Output parquet: {size_mb:.1f} MB")

    return parquet_bytes


@app.local_entrypoint()
def main(
    dry_run: bool = False,
    start_row_group: int = 0,
    batch_size: int = 4,
):
    """Run CLAP extraction and save result locally."""
    from pathlib import Path

    print("Launching Modal function...")
    result = extract_clap_embeddings.remote(
        dry_run=dry_run,
        start_row_group=start_row_group,
        batch_size=batch_size,
    )

    if isinstance(result, dict):
        print(f"\nDry run result: {result}")
        return

    output_path = (
        Path(__file__).parent / "data" / "training_data_clap_embeddings.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(result)

    size_mb = len(result) / (1024 * 1024)
    print(f"\nSaved: {output_path} ({size_mb:.1f} MB)")
    print("\n" + "=" * 60)
    print("PHASE 3.1: CLAP AUDIO EMBEDDING EXTRACTION COMPLETE")
    print("=" * 60)
