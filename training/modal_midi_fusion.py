#!/usr/bin/env python3
"""
Phase 3.2: MIDI Piano Roll + Multimodal Fusion Training on Modal

Two-phase execution:
1. CPU preprocess: Read midi_binary from media parquet -> piano rolls [128, 256]
2. GPU train: PianoRollEncoder CNN (unfrozen) + fusion MLP + regression heads

Input modalities:
    - Audio:   512-dim (CLAP, frozen precomputed)
    - MIDI:    512-dim (PianoRollEncoder CNN, trained jointly)
    - Concept: 768-dim (DeBERTa, frozen precomputed)
    - Lyric:   768-dim (DeBERTa, frozen precomputed)
    Total fusion input: 2560-dim

Usage:
    # Dry run (verify data access + piano roll conversion)
    modal run training/modal_midi_fusion.py --dry-run

    # Preprocess piano rolls only (CPU, saves to Modal volume)
    modal run training/modal_midi_fusion.py --preprocess-only

    # Full run (preprocess + train)
    modal run training/modal_midi_fusion.py

    # Skip preprocess (reuse existing piano rolls from volume)
    modal run training/modal_midi_fusion.py --skip-preprocess

    # Custom training params
    modal run training/modal_midi_fusion.py --skip-preprocess --epochs 100 --lr 5e-4
"""

import modal

app = modal.App("white-midi-fusion")

volume = modal.Volume.from_name("white-training-data", create_if_missing=True)

cpu_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "mido",
    "numpy",
    "pandas",
    "pyarrow",
    "tqdm",
    "huggingface_hub",
)

gpu_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "numpy",
    "pandas",
    "pyarrow",
    "tqdm",
    "huggingface_hub",
    "scikit-learn",
    "mido",
)

HF_REPO = "earthlyframes/white-training-data"
HF_MEDIA_PATH = "data/training_segments_media.parquet"
CACHE_DIR = "/cache"


# ============================================================================
# SHARED: Piano Roll Conversion (inlined from piano_roll_encoder.py)
# ============================================================================


def midi_bytes_to_piano_roll(
    midi_bytes: bytes,
    time_steps: int = 256,
    time_resolution_ms: float = 100.0,
    velocity_normalize: bool = True,
):
    """Convert raw MIDI file bytes to a piano roll matrix [128, time_steps]."""
    import io as _io

    import mido
    import numpy as np

    mid = mido.MidiFile(file=_io.BytesIO(midi_bytes))

    note_events = []
    for track in mid.tracks:
        abs_time_ticks = 0
        tempo = 500000  # default 120 BPM

        for msg in track:
            abs_time_ticks += msg.time
            if msg.type == "set_tempo":
                tempo = msg.tempo
            if msg.type == "note_on" and msg.velocity > 0:
                time_sec = mido.tick2second(abs_time_ticks, mid.ticks_per_beat, tempo)
                note_events.append(("on", msg.note, msg.velocity, time_sec))
            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                time_sec = mido.tick2second(abs_time_ticks, mid.ticks_per_beat, tempo)
                note_events.append(("off", msg.note, 0, time_sec))

    piano_roll = np.zeros((128, time_steps), dtype=np.float32)
    max_time_sec = time_steps * time_resolution_ms / 1000.0
    active_notes = {}
    note_events.sort(key=lambda e: e[3])

    for event_type, pitch, velocity, time_sec in note_events:
        if time_sec >= max_time_sec:
            break
        step = int(time_sec / (time_resolution_ms / 1000.0))
        step = min(step, time_steps - 1)

        if event_type == "on":
            active_notes[pitch] = (step, velocity)
        elif event_type == "off" and pitch in active_notes:
            start_step, vel = active_notes.pop(pitch)
            end_step = min(step, time_steps - 1)
            value = vel / 127.0 if velocity_normalize else 1.0
            piano_roll[pitch, start_step : end_step + 1] = value

    for pitch, (start_step, vel) in active_notes.items():
        value = vel / 127.0 if velocity_normalize else 1.0
        piano_roll[pitch, start_step:] = value

    return piano_roll


# ============================================================================
# PHASE 1: CPU PREPROCESS — MIDI binary -> piano roll matrices
# ============================================================================


@app.function(
    image=cpu_image,
    timeout=3600,
    volumes={CACHE_DIR: volume},
)
def preprocess_piano_rolls(dry_run: bool = False, start_row_group: int = 0):
    """Extract MIDI binary from media parquet, convert to piano rolls [128, 256].

    Saves piano_rolls.npz to the Modal volume for the training step.
    """
    import shutil

    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from pathlib import Path
    from tqdm import tqdm

    # --- 1. Get media parquet (cached in volume) ---
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
        downloaded = Path(local_path)
        if downloaded != cached_path:
            shutil.move(str(downloaded), str(cached_path))
        size_gb = cached_path.stat().st_size / (1024**3)
        print(f"Downloaded: {cached_path} ({size_gb:.1f} GB)")
        volume.commit()

    # --- 2. Inspect parquet ---
    pf = pq.ParquetFile(str(cached_path))
    num_row_groups = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows
    print(f"\nParquet: {total_rows} rows, {num_row_groups} row groups")

    needed_columns = ["segment_id", "midi_binary", "has_midi"]

    if dry_run:
        print("\n--- DRY RUN ---")
        rg = pf.read_row_group(0, columns=needed_columns).to_pandas()
        print(f"Row group 0: {len(rg)} rows")
        has_midi = rg["has_midi"].sum()
        has_binary = rg["midi_binary"].notna().sum()
        print(f"  has_midi=True: {has_midi}")
        print(f"  midi_binary non-null: {has_binary}")

        if has_binary > 0:
            sample = rg.loc[rg["midi_binary"].notna()].iloc[0]
            midi_bytes = sample["midi_binary"]
            print(f"  MIDI bytes length: {len(midi_bytes)}")
            roll = midi_bytes_to_piano_roll(midi_bytes)
            print(f"  Piano roll shape: {roll.shape}")
            print(f"  Non-zero pixels: {np.count_nonzero(roll)}")
            print(f"  Active pitches: {np.where(roll.any(axis=1))[0].tolist()[:10]}...")

        return {
            "status": "dry_run",
            "total_rows": total_rows,
            "row_groups": num_row_groups,
        }

    # --- 3. Process all row groups ---
    all_segment_ids = []
    all_piano_rolls = []
    all_has_midi = []
    segments_with_midi = 0
    conversion_failures = 0

    for rg_idx in range(start_row_group, num_row_groups):
        print(f"\n--- Row group {rg_idx}/{num_row_groups - 1} ---")
        rg_df = pf.read_row_group(rg_idx, columns=needed_columns).to_pandas()
        print(f"  Rows: {len(rg_df)}")

        for _, row in tqdm(rg_df.iterrows(), total=len(rg_df), desc=f"RG {rg_idx}"):
            seg_id = row["segment_id"]
            all_segment_ids.append(seg_id)
            midi_bytes = row["midi_binary"]

            if midi_bytes is not None and row.get("has_midi", False):
                try:
                    roll = midi_bytes_to_piano_roll(midi_bytes)
                    all_piano_rolls.append(roll)
                    all_has_midi.append(True)
                    segments_with_midi += 1

                    if segments_with_midi == 1:
                        print(
                            f"  First piano roll: shape={roll.shape}, "
                            f"non-zero={np.count_nonzero(roll)}"
                        )
                except Exception as e:
                    print(f"  WARNING: Failed {seg_id}: {e}")
                    all_piano_rolls.append(np.zeros((128, 256), dtype=np.float32))
                    all_has_midi.append(False)
                    conversion_failures += 1
            else:
                all_piano_rolls.append(np.zeros((128, 256), dtype=np.float32))
                all_has_midi.append(False)

        del rg_df
        print(
            f"  Progress: {len(all_segment_ids)}/{total_rows}, "
            f"{segments_with_midi} with MIDI"
        )

    # --- 4. Save to volume ---
    output_path = Path(CACHE_DIR) / "piano_rolls.npz"
    print(f"\nSaving {len(all_segment_ids)} piano rolls...")

    np.savez_compressed(
        str(output_path),
        segment_ids=np.array(all_segment_ids, dtype=object),
        piano_rolls=np.stack(all_piano_rolls),  # [N, 128, 256]
        has_midi=np.array(all_has_midi),
    )
    volume.commit()

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {output_path} ({size_mb:.1f} MB)")
    print(f"Total: {len(all_segment_ids)}")
    print(f"With MIDI: {segments_with_midi}")
    print(f"Conversion failures: {conversion_failures}")

    return {
        "status": "complete",
        "total": len(all_segment_ids),
        "with_midi": segments_with_midi,
        "failures": conversion_failures,
        "size_mb": round(size_mb, 1),
    }


# ============================================================================
# PHASE 2: GPU TRAIN — CNN + Fusion MLP + Regression Heads
# ============================================================================


@app.function(
    image=gpu_image,
    gpu="A10G",
    timeout=7200,  # 2 hours
    volumes={CACHE_DIR: volume},
)
def train_fusion(
    deberta_parquet_bytes: bytes,
    clap_parquet_bytes: bytes,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    modality_dropout: float = 0.15,
    label_smoothing: float = 0.1,
    train_split: float = 0.8,
    seed: int = 42,
):
    """Train PianoRollEncoder CNN + multimodal fusion MLP + regression heads."""
    import io

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Dataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {mem_gb:.1f} GB")

    # ----------------------------------------------------------------
    # 1. LOAD ALL DATA
    # ----------------------------------------------------------------

    # Piano rolls from volume
    volume.reload()
    npz_path = Path(CACHE_DIR) / "piano_rolls.npz"
    print(f"\nLoading piano rolls from {npz_path}...")
    npz = np.load(str(npz_path), allow_pickle=True)
    pr_segment_ids = npz["segment_ids"]
    piano_rolls = npz["piano_rolls"]  # [N, 128, 256]
    pr_has_midi = npz["has_midi"]
    print(f"  Piano rolls: {piano_rolls.shape}, {pr_has_midi.sum()} with MIDI")

    # DeBERTa embeddings (passed as bytes from local)
    # This parquet also contains ALL metadata columns (mode labels, etc.)
    print("Loading DeBERTa embeddings + metadata...")
    deberta_df = pd.read_parquet(io.BytesIO(deberta_parquet_bytes))
    print(f"  DeBERTa: {len(deberta_df)} rows, {len(deberta_df.columns)} columns")

    # CLAP embeddings (passed as bytes from local)
    print("Loading CLAP embeddings...")
    clap_df = pd.read_parquet(io.BytesIO(clap_parquet_bytes))
    print(f"  CLAP: {len(clap_df)} rows")

    # ----------------------------------------------------------------
    # 2. JOIN BY SEGMENT_ID
    # ----------------------------------------------------------------
    print("\nJoining datasets by segment_id...")

    # Build piano roll lookup: segment_id -> index in npz
    pr_lookup = {str(sid): i for i, sid in enumerate(pr_segment_ids)}

    # Build CLAP lookup: segment_id -> (embedding, has_audio)
    clap_lookup = {}
    for _, row in clap_df.iterrows():
        sid = str(row["segment_id"])
        emb = np.array(row["audio_embedding"], dtype=np.float32)
        has = bool(row.get("has_audio_embedding", False))
        clap_lookup[sid] = (emb, has)

    # DeBERTa dataframe is the base — it has embeddings + mode labels
    id_col = "segment_id" if "segment_id" in deberta_df.columns else "id"

    # Detect mode column names
    if "rainbow_color_temporal_mode" in deberta_df.columns:
        t_col = "rainbow_color_temporal_mode"
        s_col = (
            "rainbow_color_objectional_mode"
            if "rainbow_color_objectional_mode" in deberta_df.columns
            else "rainbow_color_spatial_mode"
        )
        o_col = "rainbow_color_ontological_mode"
    else:
        t_col, s_col, o_col = "temporal_mode", "spatial_mode", "ontological_mode"

    print(f"  ID column: {id_col}")
    print(f"  Mode columns: {t_col}, {s_col}, {o_col}")

    # Build aligned arrays
    TEMPORAL_MODES = ["Past", "Present", "Future"]
    SPATIAL_MODES = ["Thing", "Place", "Person"]
    ONTOLOGICAL_MODES = ["Imagined", "Forgotten", "Known"]

    segment_ids = []
    concept_embs = []
    lyric_embs = []
    has_lyrics = []
    audio_embs = []
    has_audio = []
    midi_rolls = []
    has_midi = []
    temporal_labels = []
    spatial_labels = []
    ontological_labels = []

    for _, row in deberta_df.iterrows():
        sid = str(row[id_col])
        segment_ids.append(sid)

        # Text embeddings (from DeBERTa)
        concept_embs.append(np.array(row["concept_embedding"], dtype=np.float32))
        lyric_embs.append(np.array(row["lyric_embedding"], dtype=np.float32))
        has_lyrics.append(bool(row.get("has_lyric_embedding", True)))

        # Audio embedding (from CLAP)
        if sid in clap_lookup:
            a_emb, a_has = clap_lookup[sid]
            audio_embs.append(a_emb)
            has_audio.append(a_has)
        else:
            audio_embs.append(np.zeros(512, dtype=np.float32))
            has_audio.append(False)

        # Piano roll (from preprocessed npz)
        if sid in pr_lookup:
            pr_idx = pr_lookup[sid]
            midi_rolls.append(piano_rolls[pr_idx])
            has_midi.append(bool(pr_has_midi[pr_idx]))
        else:
            midi_rolls.append(np.zeros((128, 256), dtype=np.float32))
            has_midi.append(False)

        # Mode labels (from DeBERTa df — same source as training_full)
        temporal_labels.append(row.get(t_col))
        spatial_labels.append(row.get(s_col))
        ontological_labels.append(row.get(o_col))

    print(f"  Aligned: {len(segment_ids)} segments")
    print(f"  Audio: {sum(has_audio)}/{len(has_audio)}")
    print(f"  MIDI:  {sum(has_midi)}/{len(has_midi)}")
    print(f"  Lyric: {sum(has_lyrics)}/{len(has_lyrics)}")

    # Free the large original arrays
    del piano_rolls, npz, deberta_df, clap_df

    # Convert to numpy arrays
    concept_embs = np.stack(concept_embs)  # [N, 768]
    lyric_embs = np.stack(lyric_embs)  # [N, 768]
    audio_embs = np.stack(audio_embs)  # [N, 512]
    midi_rolls = np.stack(midi_rolls)  # [N, 128, 256]
    has_audio = np.array(has_audio)
    has_midi = np.array(has_midi)
    has_lyrics = np.array(has_lyrics)

    # ----------------------------------------------------------------
    # 3. SOFT TARGETS
    # ----------------------------------------------------------------

    def to_soft_target(label, mode_list, smoothing):
        if label is None or pd.isna(label) or label == "None":
            return np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
        target = np.zeros(len(mode_list), dtype=np.float32)
        try:
            target[mode_list.index(label)] = 1.0
        except ValueError:
            return np.array([1 / len(mode_list)] * len(mode_list), dtype=np.float32)
        return (1 - smoothing) * target + smoothing * (1 / len(mode_list))

    temporal_targets = np.stack(
        [
            to_soft_target(temporal_label, TEMPORAL_MODES, label_smoothing)
            for temporal_label in temporal_labels
        ]
    )
    spatial_targets = np.stack(
        [
            to_soft_target(spatial_label, SPATIAL_MODES, label_smoothing)
            for spatial_label in spatial_labels
        ]
    )
    ontological_targets = np.stack(
        [
            to_soft_target(ontological_label, ONTOLOGICAL_MODES, label_smoothing)
            for ontological_label in ontological_labels
        ]
    )

    # Confidence: 0 for Black album (all None modes), 1 otherwise
    confidence_targets = np.array(
        [
            0.0 if (pd.isna(t) or t is None or t == "None") else 1.0
            for t in temporal_labels
        ],
        dtype=np.float32,
    )

    print("\nTarget distribution:")
    for dim, labels, modes in [
        ("Temporal", temporal_labels, TEMPORAL_MODES),
        ("Spatial", spatial_labels, SPATIAL_MODES),
        ("Ontological", ontological_labels, ONTOLOGICAL_MODES),
    ]:
        counts = {}
        for all_labels in labels:
            key = (
                str(all_labels)
                if all_labels is not None and not pd.isna(all_labels)
                else "None"
            )
            counts[key] = counts.get(key, 0) + 1
        print(f"  {dim}: {counts}")

    # ----------------------------------------------------------------
    # 4. DATASET + DATALOADER
    # ----------------------------------------------------------------

    class FusionDataset(Dataset):
        def __init__(self, indices):
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            i = self.indices[idx]
            return {
                "piano_roll": torch.from_numpy(
                    midi_rolls[i][np.newaxis, :, :]
                ),  # [1, 128, 256]
                "audio_emb": torch.from_numpy(audio_embs[i]),
                "concept_emb": torch.from_numpy(concept_embs[i]),
                "lyric_emb": torch.from_numpy(lyric_embs[i]),
                "has_audio": torch.tensor(has_audio[i], dtype=torch.bool),
                "has_midi": torch.tensor(has_midi[i], dtype=torch.bool),
                "has_lyric": torch.tensor(has_lyrics[i], dtype=torch.bool),
                "temporal_target": torch.from_numpy(temporal_targets[i]),
                "spatial_target": torch.from_numpy(spatial_targets[i]),
                "ontological_target": torch.from_numpy(ontological_targets[i]),
                "confidence_target": torch.tensor(
                    confidence_targets[i], dtype=torch.float32
                ),
            }

    # Stratified split by temporal mode (most balanced dimension)
    all_indices = np.arange(len(segment_ids))
    strat_labels = [str(temporal_label) for temporal_label in temporal_labels]
    train_idx, val_idx = train_test_split(
        all_indices,
        train_size=train_split,
        stratify=strat_labels,
        random_state=seed,
    )

    train_dataset = FusionDataset(train_idx)
    val_dataset = FusionDataset(val_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"\nTrain: {len(train_dataset)}, Val: {len(val_dataset)}")

    # ----------------------------------------------------------------
    # 5. MODEL
    # ----------------------------------------------------------------

    class PianoRollEncoder(nn.Module):
        """CNN: [batch, 1, 128, 256] -> [batch, 512]"""

        def __init__(self, output_dim=512):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.fc = nn.Sequential(
                nn.Linear(128 * 4 * 4, output_dim),
                nn.ReLU(),
            )

        def forward(self, x):
            features = self.conv(x)
            features = features.view(features.size(0), -1)
            return self.fc(features)

    class MultimodalFusionModel(nn.Module):
        """Full fusion model: piano roll CNN + null embeddings + fusion MLP + heads."""

        def __init__(self, modality_dropout=0.15):
            super().__init__()
            self.modality_dropout = modality_dropout

            # MIDI encoder (unfrozen, trains jointly)
            self.midi_encoder = PianoRollEncoder(output_dim=512)

            # Learned null embeddings for missing modalities
            self.null_audio = nn.Parameter(torch.randn(512) * 0.02)
            self.null_midi = nn.Parameter(torch.randn(512) * 0.02)
            self.null_concept = nn.Parameter(torch.randn(768) * 0.02)
            self.null_lyric = nn.Parameter(torch.randn(768) * 0.02)

            # Fusion MLP: [audio 512 + midi 512 + concept 768 + lyric 768] = 2560
            self.fusion = nn.Sequential(
                nn.Linear(2560, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
            )

            # Regression heads (softmax over 3 modes each)
            self.temporal_head = nn.Linear(512, 3)
            self.spatial_head = nn.Linear(512, 3)
            self.ontological_head = nn.Linear(512, 3)
            self.confidence_head = nn.Linear(512, 1)

        def forward(
            self,
            piano_roll,
            audio_emb,
            concept_emb,
            lyric_emb,
            has_audio,
            has_midi,
            has_lyric,
        ):
            batch_size = piano_roll.size(0)

            # Encode MIDI piano roll through CNN
            midi_emb = self.midi_encoder(piano_roll)  # [batch, 512]

            # Build modality masks (True = modality present)
            audio_mask = has_audio.unsqueeze(1)  # [batch, 1]
            midi_mask = has_midi.unsqueeze(1)
            lyric_mask = has_lyric.unsqueeze(1)

            # Modality dropout during training: randomly mask out present modalities
            if self.training and self.modality_dropout > 0:
                drop_audio = (
                    torch.rand(batch_size, 1, device=piano_roll.device)
                    < self.modality_dropout
                )
                drop_midi = (
                    torch.rand(batch_size, 1, device=piano_roll.device)
                    < self.modality_dropout
                )
                drop_lyric = (
                    torch.rand(batch_size, 1, device=piano_roll.device)
                    < self.modality_dropout
                )
                audio_mask = audio_mask & ~drop_audio
                midi_mask = midi_mask & ~drop_midi
                lyric_mask = lyric_mask & ~drop_lyric

            # Substitute null embeddings where modality is absent
            audio_emb = torch.where(
                audio_mask, audio_emb, self.null_audio.expand(batch_size, -1)
            )
            midi_emb = torch.where(
                midi_mask, midi_emb, self.null_midi.expand(batch_size, -1)
            )
            # Concept text is always present (every segment has a concept)
            lyric_emb = torch.where(
                lyric_mask, lyric_emb, self.null_lyric.expand(batch_size, -1)
            )

            # Fuse
            fused = torch.cat([audio_emb, midi_emb, concept_emb, lyric_emb], dim=-1)
            fused = self.fusion(fused)

            return {
                "temporal": F.softmax(self.temporal_head(fused), dim=-1),
                "spatial": F.softmax(self.spatial_head(fused), dim=-1),
                "ontological": F.softmax(self.ontological_head(fused), dim=-1),
                "confidence": torch.sigmoid(self.confidence_head(fused)),
            }

    model = MultimodalFusionModel(modality_dropout=modality_dropout).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    cnn_params = sum(p.numel() for p in model.midi_encoder.parameters())
    fusion_params = total_params - cnn_params
    print(f"\nModel parameters: {total_params:,}")
    print(f"  CNN encoder: {cnn_params:,}")
    print(f"  Fusion + heads: {fusion_params:,}")

    # ----------------------------------------------------------------
    # 6. TRAINING
    # ----------------------------------------------------------------

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    mse_loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_val_metrics = {}
    best_model_state = None
    patience_counter = 0
    patience_limit = 10

    print(f"\nTraining for up to {epochs} epochs...")
    print(f"  lr={lr}, batch_size={batch_size}, modality_dropout={modality_dropout}")
    print(
        f"  label_smoothing={label_smoothing}, early_stopping_patience={patience_limit}"
    )
    print()

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0
        train_batches = 0

        for batch in train_loader:
            piano_roll = batch["piano_roll"].to(device)
            audio_emb = batch["audio_emb"].to(device)
            concept_emb = batch["concept_emb"].to(device)
            lyric_emb = batch["lyric_emb"].to(device)
            h_audio = batch["has_audio"].to(device)
            h_midi = batch["has_midi"].to(device)
            h_lyric = batch["has_lyric"].to(device)

            preds = model(
                piano_roll,
                audio_emb,
                concept_emb,
                lyric_emb,
                h_audio,
                h_midi,
                h_lyric,
            )

            # KL divergence for distribution targets
            loss_t = kl_loss_fn(
                preds["temporal"].log(), batch["temporal_target"].to(device)
            )
            loss_s = kl_loss_fn(
                preds["spatial"].log(), batch["spatial_target"].to(device)
            )
            loss_o = kl_loss_fn(
                preds["ontological"].log(), batch["ontological_target"].to(device)
            )
            loss_c = mse_loss_fn(
                preds["confidence"].squeeze(-1),
                batch["confidence_target"].to(device),
            )

            loss = 0.8 * loss_t + 0.8 * loss_s + 0.8 * loss_o + 0.5 * loss_c

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches

        # --- Validate ---
        model.eval()
        val_loss = 0
        val_batches = 0
        all_preds = {"temporal": [], "spatial": [], "ontological": []}
        all_targets = {"temporal": [], "spatial": [], "ontological": []}

        with torch.no_grad():
            for batch in val_loader:
                piano_roll = batch["piano_roll"].to(device)
                audio_emb = batch["audio_emb"].to(device)
                concept_emb = batch["concept_emb"].to(device)
                lyric_emb = batch["lyric_emb"].to(device)
                h_audio = batch["has_audio"].to(device)
                h_midi = batch["has_midi"].to(device)
                h_lyric = batch["has_lyric"].to(device)

                preds = model(
                    piano_roll,
                    audio_emb,
                    concept_emb,
                    lyric_emb,
                    h_audio,
                    h_midi,
                    h_lyric,
                )

                loss_t = kl_loss_fn(
                    preds["temporal"].log(),
                    batch["temporal_target"].to(device),
                )
                loss_s = kl_loss_fn(
                    preds["spatial"].log(),
                    batch["spatial_target"].to(device),
                )
                loss_o = kl_loss_fn(
                    preds["ontological"].log(),
                    batch["ontological_target"].to(device),
                )
                loss_c = mse_loss_fn(
                    preds["confidence"].squeeze(-1),
                    batch["confidence_target"].to(device),
                )

                loss = 0.8 * loss_t + 0.8 * loss_s + 0.8 * loss_o + 0.5 * loss_c
                val_loss += loss.item()
                val_batches += 1

                for dim in ["temporal", "spatial", "ontological"]:
                    all_preds[dim].append(preds[dim].cpu())
                    all_targets[dim].append(batch[f"{dim}_target"])

        avg_val_loss = val_loss / val_batches

        # Mode accuracy: argmax of predicted distribution vs argmax of target
        mode_acc = {}
        for dim in ["temporal", "spatial", "ontological"]:
            pred_cat = torch.cat(all_preds[dim])
            targ_cat = torch.cat(all_targets[dim])
            pred_modes = pred_cat.argmax(dim=-1)
            true_modes = targ_cat.argmax(dim=-1)
            mode_acc[dim] = (pred_modes == true_modes).float().mean().item()

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f} | "
            f"T {mode_acc['temporal']:.1%} S {mode_acc['spatial']:.1%} "
            f"O {mode_acc['ontological']:.1%} | lr={current_lr:.1e}"
        )

        # Early stopping on val loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_metrics = {
                "epoch": epoch + 1,
                "val_loss": avg_val_loss,
                "temporal_acc": mode_acc["temporal"],
                "spatial_acc": mode_acc["spatial"],
                "ontological_acc": mode_acc["ontological"],
            }
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
            print(f"  ** New best model (val_loss={avg_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(
                    f"\nEarly stopping at epoch {epoch + 1} (patience={patience_limit})"
                )
                break

    # ----------------------------------------------------------------
    # 7. SAVE + RETURN
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best model at epoch {best_val_metrics['epoch']}:")
    print(f"  Val loss:        {best_val_metrics['val_loss']:.4f}")
    print(f"  Temporal mode:   {best_val_metrics['temporal_acc']:.1%}")
    print(f"  Spatial mode:    {best_val_metrics['spatial_acc']:.1%}")
    print(f"  Ontological mode:{best_val_metrics['ontological_acc']:.1%}")

    buf = io.BytesIO()
    torch.save(
        {
            "model_state_dict": best_model_state,
            "metrics": best_val_metrics,
            "config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "modality_dropout": modality_dropout,
                "label_smoothing": label_smoothing,
                "train_split": train_split,
                "seed": seed,
                "input_dims": {
                    "audio": 512,
                    "midi": 512,
                    "concept": 768,
                    "lyric": 768,
                    "fusion_input": 2560,
                },
            },
        },
        buf,
    )

    return buf.getvalue()


# ============================================================================
# LOCAL ENTRYPOINT
# ============================================================================


@app.local_entrypoint()
def main(
    dry_run: bool = False,
    preprocess_only: bool = False,
    skip_preprocess: bool = False,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
):
    """Orchestrate piano roll preprocessing and fusion training."""
    from pathlib import Path

    data_dir = Path(__file__).parent / "data"

    # --- Phase 1: Preprocess piano rolls ---
    if not skip_preprocess:
        print("=" * 60)
        print("PHASE 1: Preprocessing piano rolls (CPU)")
        print("=" * 60)
        result = preprocess_piano_rolls.remote(dry_run=dry_run)
        print(f"\nPreprocess result: {result}")

        if dry_run or preprocess_only:
            return

    # --- Phase 2: Train fusion model ---
    print("\n" + "=" * 60)
    print("PHASE 2: Training fusion model (GPU)")
    print("=" * 60)

    # Read local embedding files to pass as bytes
    deberta_path = data_dir / "training_data_with_embeddings.parquet"
    clap_path = data_dir / "training_data_clap_embeddings.parquet"

    if not deberta_path.exists():
        print(f"ERROR: DeBERTa embeddings not found: {deberta_path}")
        return
    if not clap_path.exists():
        print(f"ERROR: CLAP embeddings not found: {clap_path}")
        return

    deberta_bytes = deberta_path.read_bytes()
    clap_bytes = clap_path.read_bytes()
    print(f"DeBERTa embeddings: {len(deberta_bytes) / 1e6:.1f} MB")
    print(f"CLAP embeddings:    {len(clap_bytes) / 1e6:.1f} MB")

    model_bytes = train_fusion.remote(
        deberta_parquet_bytes=deberta_bytes,
        clap_parquet_bytes=clap_bytes,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

    # Save model locally
    output_path = data_dir / "fusion_model.pt"
    output_path.write_bytes(model_bytes)

    size_mb = len(model_bytes) / (1024 * 1024)
    print(f"\nSaved: {output_path} ({size_mb:.1f} MB)")
    print("\n" + "=" * 60)
    print("PHASE 3.2: MULTIMODAL FUSION TRAINING COMPLETE")
    print("=" * 60)
