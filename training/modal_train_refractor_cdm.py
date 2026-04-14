#!/usr/bin/env python3
"""
Refractor CDM Training on Modal.

Loads per-chunk CLAP embeddings from training/data/refractor_cdm_embeddings.npz
(produced by extract_cdm_embeddings.py), trains a small calibration MLP with
random-chunk augmentation, and exports refractor_cdm.onnx.

Two-phase workflow:
    Phase 1 (local):  python training/extract_cdm_embeddings.py
    Phase 2 (Modal):  modal run training/modal_train_refractor_cdm.py

Usage:
    # Full run (upload embeddings + train + download ONNX)
    modal run training/modal_train_refractor_cdm.py

    # Custom hyperparams
    modal run training/modal_train_refractor_cdm.py --epochs 200 --lr 5e-4

    # Audio-only (no concept embedding)
    modal run training/modal_train_refractor_cdm.py --no-concept

    # Dry run — prints dataset stats, no training
    modal run training/modal_train_refractor_cdm.py --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

app = modal.App("white-refractor-cdm")

# Tiny image — only needs torch + numpy + scikit-learn
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "numpy",
    "scikit-learn",
    "tqdm",
)

# Local embeddings file mounted into the container
REPO_ROOT = Path(__file__).parent.parent
EMBEDDINGS_PATH = REPO_ROOT / "training" / "data" / "refractor_cdm_embeddings.npz"
ONNX_OUT = REPO_ROOT / "training" / "data" / "refractor_cdm.onnx"

_COLOR_ORDER = [
    "Red",
    "Orange",
    "Yellow",
    "Green",
    "Blue",
    "Indigo",
    "Violet",
    "White",
    "Black",
]

# CHROMATIC_TARGETS — [Past/Present/Future], [Thing/Place/Person], [Imagined/Forgotten/Known]
CHROMATIC_TARGETS = {
    "Red": {
        "temporal": [0.8, 0.1, 0.1],
        "spatial": [0.8, 0.1, 0.1],
        "ontological": [0.1, 0.1, 0.8],
    },
    "Orange": {
        "temporal": [0.1, 0.8, 0.1],
        "spatial": [0.8, 0.1, 0.1],
        "ontological": [0.1, 0.1, 0.8],
    },
    "Yellow": {
        "temporal": [0.1, 0.8, 0.1],
        "spatial": [0.1, 0.8, 0.1],
        "ontological": [0.1, 0.1, 0.8],
    },
    "Green": {
        "temporal": [0.1, 0.8, 0.1],
        "spatial": [0.1, 0.8, 0.1],
        "ontological": [0.1, 0.1, 0.8],
    },
    "Blue": {
        "temporal": [0.1, 0.1, 0.8],
        "spatial": [0.1, 0.8, 0.1],
        "ontological": [0.1, 0.8, 0.1],
    },
    "Indigo": {
        "temporal": [0.1, 0.1, 0.8],
        "spatial": [0.1, 0.1, 0.8],
        "ontological": [0.1, 0.8, 0.1],
    },
    "Violet": {
        "temporal": [0.1, 0.1, 0.8],
        "spatial": [0.1, 0.1, 0.8],
        "ontological": [0.8, 0.1, 0.1],
    },
    "White": {
        "temporal": [0.33, 0.34, 0.33],
        "spatial": [0.33, 0.34, 0.33],
        "ontological": [0.33, 0.34, 0.33],
    },
    "Black": {
        "temporal": [0.1, 0.8, 0.1],
        "spatial": [0.8, 0.1, 0.1],
        "ontological": [0.8, 0.1, 0.1],
    },
}


def _color_targets(color: str) -> tuple[list, list, list]:
    t = CHROMATIC_TARGETS.get(color, CHROMATIC_TARGETS["White"])
    return t["temporal"], t["spatial"], t["ontological"]


def _top1_color(temporal, spatial, ontological) -> str:
    """Pick the color whose CHROMATIC_TARGETS best matches the predicted distributions."""
    import numpy as np

    best, best_score = "White", -1.0
    for color, targets in CHROMATIC_TARGETS.items():
        score = (
            np.dot(temporal, targets["temporal"])
            + np.dot(spatial, targets["spatial"])
            + np.dot(ontological, targets["ontological"])
        ) / 3.0
        if score > best_score:
            best_score = score
            best = color
    return best


def _stratified_split(song_ids, colors, val_frac=0.2, seed=42):
    """Return (train_mask, val_mask) over chunk indices, split by song with color stratification."""
    import numpy as np

    rng = np.random.RandomState(seed)
    unique_songs = list(
        dict.fromkeys(zip(song_ids, colors))
    )  # preserve order, deduplicate
    val_songs: set[str] = set()

    # Group songs by color, pick ~val_frac per color
    by_color: dict[str, list[str]] = {}
    for sid, col in unique_songs:
        by_color.setdefault(col, []).append(sid)

    for col, sids in by_color.items():
        n_val = max(1, round(len(sids) * val_frac))
        chosen = rng.choice(sids, size=n_val, replace=False)
        val_songs.update(chosen)

    train_mask = np.array([sid not in val_songs for sid in song_ids])
    val_mask = ~train_mask
    return train_mask, val_mask


@app.function(
    image=image,
    timeout=1800,
)
def train(
    embeddings_bytes: bytes,
    epochs: int = 150,
    lr: float = 1e-3,
    batch_size: int = 64,
    use_concept: bool = True,
    dry_run: bool = False,
    seed: int = 42,
) -> bytes:
    """Train the Refractor CDM MLP and return the ONNX bytes."""
    import io

    import numpy as np
    import torch
    import torch.nn as nn

    # ------------------------------------------------------------------
    # Load embeddings
    # ------------------------------------------------------------------
    data = np.load(io.BytesIO(embeddings_bytes), allow_pickle=False)
    clap_embs = data["clap_embs"].astype(np.float32)  # (N, 512)
    concept_embs = data["concept_embs"].astype(np.float32)  # (N, 768)
    colors = data["colors"]  # (N,) str
    song_ids = data["song_ids"]  # (N,) str

    N = len(clap_embs)
    unique_songs = len(set(song_ids))
    unique_colors = sorted(set(colors))
    color_counts = {c: int((colors == c).sum()) for c in unique_colors}

    print(f"\nDataset: {N} chunks from {unique_songs} songs")
    print(f"Colors: {color_counts}")
    print(f"use_concept={use_concept}  epochs={epochs}  lr={lr}  batch={batch_size}\n")

    if dry_run:
        print("Dry run — exiting before training.")
        return b""

    # ------------------------------------------------------------------
    # Train/val split
    # ------------------------------------------------------------------
    train_mask, val_mask = _stratified_split(song_ids, colors, val_frac=0.2, seed=seed)
    print(f"Train: {train_mask.sum()} chunks  |  Val: {val_mask.sum()} chunks")
    val_colors = colors[val_mask]
    for c in unique_colors:
        n_val = int((val_colors == c).sum())
        print(f"  {c:<8} val={n_val}")
    print()

    clap_train, concept_train, colors_train = (
        clap_embs[train_mask],
        concept_embs[train_mask],
        colors[train_mask],
    )
    clap_val, concept_val, colors_val = (
        clap_embs[val_mask],
        concept_embs[val_mask],
        colors[val_mask],
    )

    # Build soft label arrays
    def make_labels(color_arr):
        T = np.zeros((len(color_arr), 3), dtype=np.float32)
        S = np.zeros((len(color_arr), 3), dtype=np.float32)
        Ont = np.zeros((len(color_arr), 3), dtype=np.float32)
        for i, c in enumerate(color_arr):
            t, s, o = _color_targets(c)
            T[i] = t
            S[i] = s
            Ont[i] = o
        return T, S, Ont

    T_train, S_train, O_train = make_labels(colors_train)
    T_val, S_val, O_val = make_labels(colors_val)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    input_dim = 512 + (768 if use_concept else 0)

    class CDMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            self.t_head = nn.Linear(128, 3)
            self.s_head = nn.Linear(128, 3)
            self.o_head = nn.Linear(128, 3)

        def forward(self, x):
            h = self.shared(x)
            return (
                torch.softmax(self.t_head(h), dim=-1),
                torch.softmax(self.s_head(h), dim=-1),
                torch.softmax(self.o_head(h), dim=-1),
            )

    torch.manual_seed(seed)
    model = CDMModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    def to_tensor(arr):
        return torch.FloatTensor(arr.tolist())

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_mean_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        # Shuffle training data each epoch (random-chunk augmentation)
        idx = np.random.permutation(len(clap_train))
        clap_s, concept_s = clap_train[idx], concept_train[idx]
        T_s, S_s, O_s = T_train[idx], S_train[idx], O_train[idx]

        train_loss = 0.0
        n_batches = 0
        for start in range(0, len(clap_s), batch_size):
            end = start + batch_size
            x_clap = to_tensor(clap_s[start:end])
            x_concept = to_tensor(concept_s[start:end])
            x = torch.cat([x_clap, x_concept], dim=-1) if use_concept else x_clap

            t_pred, s_pred, o_pred = model(x)
            loss = (
                mse(t_pred, to_tensor(T_s[start:end]))
                + mse(s_pred, to_tensor(S_s[start:end]))
                + mse(o_pred, to_tensor(O_s[start:end]))
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        # Val accuracy
        model.eval()
        with torch.no_grad():
            x_clap_v = to_tensor(clap_val)
            x_concept_v = to_tensor(concept_val)
            x_v = (
                torch.cat([x_clap_v, x_concept_v], dim=-1) if use_concept else x_clap_v
            )
            t_v, s_v, o_v = model(x_v)

        t_np = t_v.numpy()
        s_np = s_v.numpy()
        o_np = o_v.numpy()

        per_color_correct: dict[str, list] = {c: [] for c in unique_colors}
        for i, true_c in enumerate(colors_val):
            pred_c = _top1_color(t_np[i], s_np[i], o_np[i])
            per_color_correct[true_c].append(pred_c == true_c)

        color_accs = {
            c: (sum(v) / len(v) if v else 0.0) for c, v in per_color_correct.items()
        }
        mean_acc = float(np.mean(list(color_accs.values())))

        if mean_acc > best_mean_acc:
            best_mean_acc = mean_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            overall = sum(sum(v) for v in per_color_correct.values()) / max(
                1, len(colors_val)
            )
            print(
                f"Epoch {epoch:4d}  loss={train_loss/n_batches:.4f}  "
                f"val_acc={overall:.1%}  mean_color_acc={mean_acc:.1%}  "
                f"best={best_mean_acc:.1%}"
            )

    # ------------------------------------------------------------------
    # Final validation with best checkpoint
    # ------------------------------------------------------------------
    model.load_state_dict(best_state)
    model.eval()
    print(f"\nBest mean color accuracy: {best_mean_acc:.1%}")
    print("\nPer-color val accuracy (best checkpoint):")
    with torch.no_grad():
        x_v = (
            torch.cat([to_tensor(clap_val), to_tensor(concept_val)], dim=-1)
            if use_concept
            else to_tensor(clap_val)
        )
        t_v, s_v, o_v = model(x_v)
    t_np, s_np, o_np = t_v.numpy(), s_v.numpy(), o_v.numpy()

    per_color_correct = {c: [] for c in unique_colors}
    for i, true_c in enumerate(colors_val):
        pred_c = _top1_color(t_np[i], s_np[i], o_np[i])
        per_color_correct[true_c].append(pred_c == true_c)

    total_correct = 0
    for c in unique_colors:
        v = per_color_correct[c]
        n_correct = sum(v)
        total_correct += n_correct
        acc = n_correct / len(v) if v else 0.0
        print(f"  {c:<8} {n_correct}/{len(v)}  {acc:.1%}")

    overall_acc = total_correct / len(colors_val)
    print(f"\n  OVERALL  {total_correct}/{len(colors_val)}  {overall_acc:.1%}")
    if overall_acc < 0.70:
        print(
            "\n  WARNING: accuracy below 70% threshold — consider more data or epochs"
        )

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------
    dummy_clap = torch.zeros(1, 512)
    dummy_concept = torch.zeros(1, 768)

    if use_concept:
        dummy_in = torch.cat([dummy_clap, dummy_concept], dim=-1)
        input_names = ["input"]
    else:
        dummy_in = dummy_clap
        input_names = ["input"]

    buf = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_in,
            buf,
            input_names=input_names,
            output_names=["temporal", "spatial", "ontological"],
            dynamic_axes={
                "input": {0: "batch"},
                "temporal": {0: "batch"},
                "spatial": {0: "batch"},
                "ontological": {0: "batch"},
            },
            opset_version=14,
        )
    onnx_bytes = buf.getvalue()
    print(f"\nONNX size: {len(onnx_bytes)/1024:.1f} KB")
    return onnx_bytes


@app.local_entrypoint()
def main(
    epochs: int = 150,
    lr: float = 1e-3,
    batch_size: int = 64,
    no_concept: bool = False,
    dry_run: bool = False,
    seed: int = 42,
    onnx_out: str = "",
):
    if not EMBEDDINGS_PATH.exists():
        print(f"ERROR: embeddings not found at {EMBEDDINGS_PATH}", file=sys.stderr)
        print("Run: python training/extract_cdm_embeddings.py", file=sys.stderr)
        sys.exit(1)

    use_concept = not no_concept
    embeddings_bytes = EMBEDDINGS_PATH.read_bytes()
    print(
        f"Uploading embeddings ({len(embeddings_bytes) / 1e6:.1f} MB) …  "
        f"epochs={epochs}  lr={lr}  use_concept={use_concept}"
    )

    onnx_bytes = train.remote(
        embeddings_bytes=embeddings_bytes,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        use_concept=use_concept,
        dry_run=dry_run,
        seed=seed,
    )

    if not onnx_bytes:
        return

    out = Path(onnx_out) if onnx_out else ONNX_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(onnx_bytes)
    print(f"\nRefractor CDM saved → {out}")
    print("Next: run validate_mix_scoring.py to confirm accuracy on full catalog.")
