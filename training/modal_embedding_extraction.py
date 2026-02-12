#!/usr/bin/env python3
"""
Phase 3.0: DeBERTa Embedding Extraction on Modal

Replaces the RunPod notebook with a Modal serverless GPU job.
Pulls metadata from HuggingFace, runs DeBERTa-v3-base, saves embeddings locally.

Usage:
    # Dry run (check data loads correctly, no GPU)
    modal run training/modal_embedding_extraction.py --dry-run

    # Full extraction
    modal run training/modal_embedding_extraction.py

    # Custom batch size (reduce if OOM)
    modal run training/modal_embedding_extraction.py --batch-size 16
"""

import modal

app = modal.App("white-deberta-embeddings")

# Image with all dependencies pre-installed
gpu_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers",
    "pandas",
    "pyarrow",
    "numpy",
    "tqdm",
    "sentencepiece",
    "protobuf",
    "safetensors",
    "datasets",
    "huggingface_hub",
)

MODEL_NAME = "microsoft/deberta-v3-base"
HF_REPO = "earthlyframes/white-training-data"
# Direct path to the 0.6 MB metadata parquet (avoids listing the full repo tree
# which times out due to the 15 GB media file)
HF_PARQUET_PATH = "training_full/train-00000-of-00001.parquet"
HIDDEN_SIZE = 768
MAX_LENGTH = 512


@app.function(
    image=gpu_image,
    gpu="A10G",
    timeout=1800,  # 30 min max
    retries=0,
)
def extract_embeddings(batch_size: int = 32, dry_run: bool = False):
    """Extract DeBERTa embeddings for all training segments."""
    import torch
    import numpy as np
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, AutoModel
    from tqdm import tqdm

    # --- 1. Load data from HuggingFace ---
    # Download the specific parquet file directly (avoids repo tree listing timeout)
    print(f"Downloading {HF_PARQUET_PATH} from {HF_REPO}...")
    local_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_PARQUET_PATH,
        repo_type="dataset",
    )
    df = pd.read_parquet(local_path)
    print(f"Loaded {len(df)} segments, {len(df.columns)} columns")

    # Check text columns
    concept_texts = df["concept"].fillna("").astype(str).tolist()
    lyric_texts = df["lyric_text"].fillna("").astype(str).tolist()

    concept_non_empty = sum(1 for t in concept_texts if t.strip())
    lyric_non_empty = sum(1 for t in lyric_texts if t.strip())
    print(f"Concept texts: {concept_non_empty}/{len(concept_texts)} non-empty")
    print(f"Lyric texts:   {lyric_non_empty}/{len(lyric_texts)} non-empty")

    if dry_run:
        print("\n--- DRY RUN: data loaded successfully, skipping GPU work ---")
        print(f"Columns: {list(df.columns)}")
        if "rainbow_color" in df.columns:
            print(f"Colors: {sorted(df['rainbow_color'].unique())}")
        return {"status": "dry_run", "segments": len(df)}

    # --- 2. Load model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Loading model...")
    model = AutoModel.from_pretrained(MODEL_NAME, use_safetensors=True)
    model = model.to(device)
    model.eval()
    print(f"Model loaded: hidden_size={model.config.hidden_size}")

    # --- 3. Embedding extraction ---
    def _embed(texts, label):
        embeddings = []
        num_batches = (len(texts) + batch_size - 1) // batch_size
        print(
            f"\nExtracting {label} embeddings: {len(texts)} texts, {num_batches} batches"
        )

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc=label):
                batch = texts[i : i + batch_size]
                batch_emb = np.zeros((len(batch), HIDDEN_SIZE), dtype=np.float32)
                non_empty = [j for j, t in enumerate(batch) if t.strip()]

                if non_empty:
                    ne_texts = [batch[j] for j in non_empty]
                    encoded = tokenizer(
                        ne_texts,
                        max_length=MAX_LENGTH,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    ids = encoded["input_ids"].to(device)
                    mask = encoded["attention_mask"].to(device)
                    out = model(input_ids=ids, attention_mask=mask)
                    ne_emb = out.last_hidden_state[:, 0, :].cpu().numpy()

                    for k, j in enumerate(non_empty):
                        batch_emb[j] = ne_emb[k]

                    del ids, mask, out, ne_emb
                    torch.cuda.empty_cache()

                embeddings.extend(batch_emb)

        print(f"  Done: {len(embeddings)} embeddings")
        return embeddings

    concept_embeddings = _embed(concept_texts, "concept")
    lyric_embeddings = _embed(lyric_texts, "lyric")

    zero_lyrics = sum(1 for e in lyric_embeddings if np.allclose(e, 0))
    print(
        f"\nZero-vector lyric embeddings (instrumental): {zero_lyrics}/{len(lyric_embeddings)}"
    )

    # --- 4. Build output dataframe ---
    df["concept_embedding"] = concept_embeddings
    df["lyric_embedding"] = lyric_embeddings
    df["has_lyric_embedding"] = [bool(t.strip()) for t in lyric_texts]

    print(f"\nhas_lyric_embedding: {df['has_lyric_embedding'].sum()}/{len(df)}")

    # --- 5. Serialize to bytes for return ---
    import io

    buf = io.BytesIO()
    df.to_parquet(buf)
    parquet_bytes = buf.getvalue()
    size_mb = len(parquet_bytes) / (1024 * 1024)
    print(f"\nOutput parquet: {size_mb:.1f} MB")

    return parquet_bytes


@app.local_entrypoint()
def main(batch_size: int = 32, dry_run: bool = False):
    """Run embedding extraction and save result locally."""
    from pathlib import Path

    print("Launching Modal function...")
    result = extract_embeddings.remote(batch_size=batch_size, dry_run=dry_run)

    if isinstance(result, dict):
        # Dry run
        print(f"\nDry run result: {result}")
        return

    # Save parquet locally
    output_path = (
        Path(__file__).parent / "data" / "training_data_with_embeddings.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(result)

    size_mb = len(result) / (1024 * 1024)
    print(f"\nSaved: {output_path} ({size_mb:.1f} MB)")
    print("\n" + "=" * 60)
    print("PHASE 3.0: EMBEDDING EXTRACTION COMPLETE")
    print("=" * 60)
