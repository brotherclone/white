#!/usr/bin/env python3
"""
Phase 4: Regression Tasks Training Script
Uses runtime soft target generation - no parquet changes needed!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List
import wandb
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "data": {
        "parquet_path": "/mnt/user-data/uploads/base_manifest_db.parquet",
        "embeddings_path": None,  # Will load from your 69GB file if needed
        "label_smoothing": 0.1,
        "train_split": 0.8,
        "batch_size": 32,
    },
    "model": {
        "embedding_dim": 768,  # Adjust based on your Phase 2 encoder
        "hidden_dim": 256,
        "dropout": 0.1,
    },
    "training": {
        "epochs": 50,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "loss_weights": {
            "classification": 1.0,
            "temporal_regression": 0.8,
            "spatial_regression": 0.8,
            "ontological_regression": 0.8,
            "confidence_regression": 0.5,
        },
        "early_stopping_patience": 10,
    },
    "validation": {
        "confidence_threshold": 0.7,
        "dominant_threshold": 0.6,
        "hybrid_threshold": 0.15,
        "diffuse_threshold": 0.2,
    },
    "wandb": {
        "project": "White",
        "name": "phase4-regression-v1",
    },
}


# ============================================================================
# SOFT TARGET GENERATION
# ============================================================================


class SoftTargetGenerator:
    """Converts discrete Rainbow Table labels to continuous regression targets"""

    TEMPORAL_MODES = ["Past", "Present", "Future"]
    SPATIAL_MODES = ["Thing", "Place", "Person"]
    ONTOLOGICAL_MODES = ["Imagined", "Forgotten", "Known"]

    def __init__(self, label_smoothing: float = 0.1):
        self.smoothing = label_smoothing

    def to_soft_target(self, label: str, mode_list: List[str]) -> np.ndarray:
        """Convert discrete label to smoothed soft target"""
        if label is None or pd.isna(label) or label == "None":
            # Black Album (None_None_None) - uniform distribution
            return np.array([1 / 3, 1 / 3, 1 / 3])

        # Create one-hot
        target = np.zeros(len(mode_list))
        try:
            target[mode_list.index(label)] = 1.0
        except ValueError:
            # Fallback to uniform if label not recognized
            return np.array([1 / len(mode_list)] * len(mode_list))

        # Apply label smoothing: (1-α)*one_hot + α*uniform
        smoothed = (1 - self.smoothing) * target + self.smoothing * (1 / len(mode_list))
        return smoothed

    def generate_targets(self, row: pd.Series) -> Dict[str, np.ndarray]:
        """Generate all regression targets for a segment"""
        # Get discrete labels
        temporal_label = row.get("rainbow_color_temporal_mode")
        spatial_label = row.get("rainbow_color_objectional_mode")
        ontological_label = row.get("rainbow_color_ontological_mode")

        # Convert to soft targets
        temporal = self.to_soft_target(temporal_label, self.TEMPORAL_MODES)
        spatial = self.to_soft_target(spatial_label, self.SPATIAL_MODES)
        ontological = self.to_soft_target(ontological_label, self.ONTOLOGICAL_MODES)

        # Confidence: 1.0 for clear assignments, 0.0 for Black Album
        is_black = all(
            pd.isna(x) or x is None or x == "None"
            for x in [temporal_label, spatial_label, ontological_label]
        )
        confidence = np.array([0.0 if is_black else 1.0])

        return {
            "temporal": temporal,
            "spatial": spatial,
            "ontological": ontological,
            "confidence": confidence,
        }


# ============================================================================
# DATASET
# ============================================================================


class RainbowTableRegressionDataset(Dataset):
    """Dataset for regression training with runtime soft target generation"""

    def __init__(self, parquet_path: str, label_smoothing: float = 0.1):
        self.df = pd.read_parquet(parquet_path)
        self.target_generator = SoftTargetGenerator(label_smoothing)

        print(f"Loaded {len(self.df)} segments")
        print(f"  Albums: {self.df['rainbow_color'].value_counts().to_dict()}")
        print(
            f"  Temporal: {self.df['rainbow_color_temporal_mode'].value_counts().to_dict()}"
        )
        print(
            f"  Spatial: {self.df['rainbow_color_objectional_mode'].value_counts().to_dict()}"
        )
        print(
            f"  Ontological: {self.df['rainbow_color_ontological_mode'].value_counts().to_dict()}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Generate soft targets
        targets = self.target_generator.generate_targets(row)

        # For now, use random embeddings as placeholder
        # TODO: Load actual embeddings from your 69GB file
        embedding = torch.randn(768)  # Replace with real embeddings

        return {
            "embedding": embedding,
            "temporal_target": torch.tensor(targets["temporal"], dtype=torch.float32),
            "spatial_target": torch.tensor(targets["spatial"], dtype=torch.float32),
            "ontological_target": torch.tensor(
                targets["ontological"], dtype=torch.float32
            ),
            "confidence_target": torch.tensor(
                targets["confidence"], dtype=torch.float32
            ),
            "segment_id": row["id"],
        }


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================


class RainbowTableRegressionHead(nn.Module):
    """Regression head for continuous ontological mode prediction"""

    def __init__(
        self, embedding_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.1
    ):
        super().__init__()

        # Shared transformation
        self.shared = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Separate heads for each ontological dimension
        self.temporal_head = nn.Linear(hidden_dim, 3)  # Past/Present/Future
        self.spatial_head = nn.Linear(hidden_dim, 3)  # Thing/Place/Person
        self.ontological_head = nn.Linear(hidden_dim, 3)  # Imagined/Forgotten/Known
        self.confidence_head = nn.Linear(hidden_dim, 1)  # Chromatic confidence

    def forward(self, embeddings):
        """
        Args:
            embeddings: [batch, embedding_dim] tensor

        Returns:
            dict with keys: temporal, spatial, ontological, confidence
        """
        x = self.shared(embeddings)

        return {
            "temporal": F.softmax(
                self.temporal_head(x), dim=-1
            ),  # [batch, 3], sums to 1
            "spatial": F.softmax(self.spatial_head(x), dim=-1),  # [batch, 3], sums to 1
            "ontological": F.softmax(
                self.ontological_head(x), dim=-1
            ),  # [batch, 3], sums to 1
            "confidence": torch.sigmoid(
                self.confidence_head(x)
            ),  # [batch, 1], in [0,1]
        }


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================


class MultiTaskRegressionLoss(nn.Module):
    """Combined loss for temporal, spatial, ontological regression + confidence"""

    def __init__(self, weights: Dict[str, float]):
        super().__init__()
        self.weights = weights
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict with temporal, spatial, ontological, confidence tensors
            targets: dict with temporal_target, spatial_target, etc.
        """
        losses = {}

        # KL divergence for probability distributions (better than MSE for softmax outputs)
        losses["temporal"] = self.kl_div(
            predictions["temporal"].log(), targets["temporal_target"]
        )
        losses["spatial"] = self.kl_div(
            predictions["spatial"].log(), targets["spatial_target"]
        )
        losses["ontological"] = self.kl_div(
            predictions["ontological"].log(), targets["ontological_target"]
        )

        # MSE for confidence score
        losses["confidence"] = self.mse(
            predictions["confidence"], targets["confidence_target"]
        )

        # Weighted sum
        total_loss = (
            self.weights["temporal_regression"] * losses["temporal"]
            + self.weights["spatial_regression"] * losses["spatial"]
            + self.weights["ontological_regression"] * losses["ontological"]
            + self.weights["confidence_regression"] * losses["confidence"]
        )

        return total_loss, losses


# ============================================================================
# EVALUATION METRICS
# ============================================================================


def compute_regression_metrics(predictions, targets, prefix="val"):
    """Compute comprehensive regression metrics"""
    metrics = {}

    for dim in ["temporal", "spatial", "ontological"]:
        pred = predictions[dim].cpu().numpy()  # [batch, 3]
        targ = targets[f"{dim}_target"].cpu().numpy()  # [batch, 3]

        # Per-dimension metrics
        mae = mean_absolute_error(targ.flatten(), pred.flatten())

        # Correlation for each of the 3 positions
        for i, mode in enumerate(["0", "1", "2"]):  # Positions in softmax
            if len(targ[:, i]) > 1:  # Need at least 2 points
                pearson_r, _ = pearsonr(targ[:, i], pred[:, i])
                metrics[f"{prefix}/{dim}_{mode}_pearson"] = pearson_r

        metrics[f"{prefix}/{dim}_mae"] = mae

    # Confidence metrics
    conf_pred = predictions["confidence"].cpu().numpy().flatten()
    conf_targ = targets["confidence_target"].cpu().numpy().flatten()
    metrics[f"{prefix}/confidence_mae"] = mean_absolute_error(conf_targ, conf_pred)
    if len(conf_targ) > 1:
        pearson_r, _ = pearsonr(conf_targ, conf_pred)
        metrics[f"{prefix}/confidence_pearson"] = pearson_r

    # Album prediction accuracy (from continuous scores)
    pred_albums = predict_albums_from_scores(predictions)
    true_albums = predict_albums_from_scores(targets)  # From soft targets
    accuracy = (pred_albums == true_albums).float().mean().item()
    metrics[f"{prefix}/album_accuracy"] = accuracy

    return metrics


def predict_albums_from_scores(outputs):
    """Predict album from continuous ontological scores"""
    # Get argmax for each dimension
    temporal = outputs["temporal"].argmax(dim=-1)  # 0=Past, 1=Present, 2=Future
    spatial = outputs["spatial"].argmax(dim=-1)  # 0=Thing, 1=Place, 2=Person
    ontological = outputs["ontological"].argmax(
        dim=-1
    )  # 0=Imagined, 1=Forgotten, 2=Known

    # Map to albums (based on your data analysis)
    # Past_Thing_Imagined=Orange, Present_Person_Forgotten=Blue, etc.
    album_map = {
        (0, 0, 0): 0,  # Past_Thing_Imagined → Orange
        (0, 0, 2): 1,  # Past_Thing_Known → Red
        (2, 1, 0): 2,  # Future_Place_Imagined → Yellow
        (2, 1, 1): 3,  # Future_Place_Forgotten → Green
        (1, 2, 1): 4,  # Present_Person_Forgotten → Blue
        # Add Indigo mapping when you have it
    }

    albums = []
    for t, s, o in zip(temporal, spatial, ontological):
        key = (t.item(), s.item(), o.item())
        albums.append(album_map.get(key, 5))  # 5 = Black/Unknown

    return torch.tensor(albums)


# ============================================================================
# TRAINING LOOP
# ============================================================================


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_losses = {k: 0 for k in ["temporal", "spatial", "ontological", "confidence"]}

    for batch in dataloader:
        # Move to device
        embeddings = batch["embedding"].to(device)
        targets = {k: v.to(device) for k, v in batch.items() if "target" in k}

        # Forward pass
        predictions = model(embeddings)
        loss, component_losses = criterion(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Track losses
        total_loss += loss.item()
        for k, v in component_losses.items():
            all_losses[k] += v.item()

    n_batches = len(dataloader)
    return {
        "total_loss": total_loss / n_batches,
        **{f"{k}_loss": v / n_batches for k, v in all_losses.items()},
    }


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_predictions = {
        k: [] for k in ["temporal", "spatial", "ontological", "confidence"]
    }
    all_targets = {
        k: []
        for k in [
            "temporal_target",
            "spatial_target",
            "ontological_target",
            "confidence_target",
        ]
    }

    with torch.no_grad():
        for batch in dataloader:
            embeddings = batch["embedding"].to(device)
            targets = {k: v.to(device) for k, v in batch.items() if "target" in k}

            predictions = model(embeddings)
            loss, _ = criterion(predictions, targets)

            total_loss += loss.item()

            # Collect predictions and targets
            for k in all_predictions.keys():
                all_predictions[k].append(predictions[k])
            for k in all_targets.keys():
                all_targets[k].append(targets[k])

    # Concatenate all batches
    predictions_cat = {k: torch.cat(v) for k, v in all_predictions.items()}
    targets_cat = {k: torch.cat(v) for k, v in all_targets.items()}

    # Compute metrics
    metrics = compute_regression_metrics(predictions_cat, targets_cat)
    metrics["val/total_loss"] = total_loss / len(dataloader)

    return metrics


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================


def main():
    """Run Phase 4 regression training"""

    # Initialize wandb
    wandb.init(
        project=CONFIG["wandb"]["project"], name=CONFIG["wandb"]["name"], config=CONFIG
    )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print("\n📊 Loading dataset...")
    dataset = RainbowTableRegressionDataset(
        parquet_path=CONFIG["data"]["parquet_path"],
        label_smoothing=CONFIG["data"]["label_smoothing"],
    )

    # Split train/val
    train_size = int(len(dataset) * CONFIG["data"]["train_split"])
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["data"]["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["data"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    print(f"  Train: {len(train_dataset)} segments")
    print(f"  Val: {len(val_dataset)} segments")

    # Initialize model
    print("\n🏗️  Initializing model...")
    model = RainbowTableRegressionHead(
        embedding_dim=CONFIG["model"]["embedding_dim"],
        hidden_dim=CONFIG["model"]["hidden_dim"],
        dropout=CONFIG["model"]["dropout"],
    ).to(device)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup training
    criterion = MultiTaskRegressionLoss(CONFIG["training"]["loss_weights"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["training"]["lr"],
        weight_decay=CONFIG["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # Training loop
    print("\n🚀 Starting training...\n")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(CONFIG["training"]["epochs"]):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Log to wandb
        wandb.log({**train_metrics, **val_metrics, "epoch": epoch})

        # Update learning rate
        scheduler.step(val_metrics["val/total_loss"])

        # Print progress
        print(f"Epoch {epoch + 1}/{CONFIG['training']['epochs']}")
        print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
        print(f"  Val Loss: {val_metrics['val/total_loss']:.4f}")
        print(f"  Album Accuracy: {val_metrics['val/album_accuracy']:.3f}")

        # Early stopping
        if val_metrics["val/total_loss"] < best_val_loss:
            best_val_loss = val_metrics["val/total_loss"]
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "phase4_regression_best.pt")
            print("  ✅ New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["training"]["early_stopping_patience"]:
                print(f"\n⏹️  Early stopping triggered (patience={patience_counter})")
                break

        print()

    print("\n✅ Training complete!")
    print(f"   Best val loss: {best_val_loss:.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()
