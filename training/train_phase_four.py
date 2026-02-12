#!/usr/bin/env python3
"""
RunPod Phase 4 Regression Training Launcher

Trains Rainbow Table ontological regression model on RunPod GPU.

Usage:
    python runpod_train_phase4.py
    python runpod_train_phase4.py --skip-deps  # If deps already installed
"""

import subprocess
import sys
from pathlib import Path
import shutil


def install_dependencies():
    """Install required packages for Phase 4."""
    print("=" * 80)
    print("INSTALLING PHASE 4 DEPENDENCIES")
    print("=" * 80)

    packages = [
        "torch",
        "transformers",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "scipy",
        "pyyaml",
        "tqdm",
        "wandb",
        "matplotlib",
        "seaborn",
        "numpy",
    ]

    print(f"Installing {len(packages)} packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])
    print("✅ Dependencies installed.\n")


def check_gpu():
    """Verify GPU availability."""
    print("=" * 80)
    print("CHECKING GPU")
    print("=" * 80)

    import torch

    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA not available!")
        print("   This script requires a GPU.")
        sys.exit(1)

    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()


def check_data():
    """Verify training data exists."""
    print("=" * 80)
    print("CHECKING DATA")
    print("=" * 80)

    data_dir = Path("/workspace/data")

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("\nExpected structure:")
        print("  /workspace/data/")
        print("    ├── base_manifest_db.parquet")
        print("    └── training_segments_media.parquet (optional, 69GB)")
        sys.exit(1)

    # Check base manifest (required)
    manifest = data_dir / "base_manifest_db.parquet"
    if not manifest.exists():
        print(f"❌ Required file missing: {manifest}")
        sys.exit(1)

    manifest_size_mb = manifest.stat().st_size / (1024 * 1024)
    print(f"✅ base_manifest_db.parquet ({manifest_size_mb:.1f} MB)")

    # Check embeddings (optional but recommended)
    embedded = data_dir / "training_segments_media.parquet"
    if embedded.exists():
        size_gb = embedded.stat().st_size / (1024**3)
        print(f"✅ training_segments_media.parquet ({size_gb:.1f} GB)")
        print("   Using pre-computed embeddings")
    else:
        print("⚠️  training_segments_media.parquet not found")
        print("   Will compute embeddings on-the-fly (slower)")

    print()


def setup_training_code():
    """Set up training code in workspace."""
    print("=" * 80)
    print("SETTING UP TRAINING CODE")
    print("=" * 80)

    workspace = Path("/workspace")
    training_dir = workspace / "phase4_training"

    # Create training directory
    training_dir.mkdir(exist_ok=True)
    print(f"✅ Training directory: {training_dir}")

    # Copy config (should already be in repo)
    repo_config = Path(__file__).parent / "config_regression.yml"
    if repo_config.exists():
        shutil.copy(repo_config, training_dir / "config_regression.yml")
        print("✅ Copied config_regression.yml")
    else:
        print("⚠️  config_regression.yml not found in repo")
        print(f"   Expected at: {repo_config}")

    print()
    return training_dir


def create_phase4_trainer(training_dir: Path):
    """Create Phase 4 training script adapted for RunPod."""
    print("=" * 80)
    print("CREATING PHASE 4 TRAINER")
    print("=" * 80)

    trainer_code = '''#!/usr/bin/env python3
"""
Phase 4 Regression Training - RunPod Adapted
Trains Rainbow Table ontological regression model with pre-computed embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import wandb
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


class PrecomputedEmbeddingLoader:
    """Loads pre-computed embeddings from parquet file."""

    def __init__(self, parquet_path: Path, id_column: str = "id", embedding_column: str = "embedding"):
        self.parquet_path = Path(parquet_path)
        self.id_column = id_column
        self.embedding_column = embedding_column

        print(f"Loading embeddings from {self.parquet_path}...")
        self.df = pd.read_parquet(self.parquet_path)

        if self.embedding_column not in self.df.columns:
            raise ValueError(f"Embedding column '{self.embedding_column}' not found")

        # Create index for fast lookup
        if self.id_column in self.df.columns:
            self._id_to_idx = {str(row[self.id_column]): idx for idx, row in self.df.iterrows()}
        else:
            self._id_to_idx = None

        sample = self.df[self.embedding_column].iloc[0]
        self.embedding_dim = len(np.array(sample))
        print(f"Loaded {len(self.df)} embeddings (dim={self.embedding_dim})")

    def get_embedding_by_idx(self, idx: int) -> np.ndarray:
        return np.array(self.df[self.embedding_column].iloc[idx], dtype=np.float32)

    def __len__(self):
        return len(self.df)


def find_embedding_file(data_dir: Path) -> Optional[Path]:
    """Find the best available embedding file."""
    candidates = ["training_data_with_embeddings.parquet", "training_segments_media.parquet"]
    for filename in candidates:
        path = data_dir / filename
        if path.exists():
            size_gb = path.stat().st_size / (1024**3)
            print(f"Found embedding file: {path} ({size_gb:.2f} GB)")
            return path
    return None


class SoftTargetGenerator:
    """Generates soft regression targets from discrete labels"""

    TEMPORAL_MODES = ['Past', 'Present', 'Future']
    SPATIAL_MODES = ['Thing', 'Place', 'Person']
    ONTOLOGICAL_MODES = ['Imagined', 'Forgotten', 'Known']

    def __init__(self, label_smoothing: float = 0.1):
        self.smoothing = label_smoothing

    def to_soft_target(self, label: str, mode_list: List[str]) -> np.ndarray:
        """Convert discrete label to smoothed soft target"""
        if label is None or pd.isna(label) or label == 'None':
            return np.array([1/3, 1/3, 1/3])

        target = np.zeros(len(mode_list))
        try:
            target[mode_list.index(label)] = 1.0
        except ValueError:
            return np.array([1/len(mode_list)] * len(mode_list))

        smoothed = (1 - self.smoothing) * target + self.smoothing * (1/len(mode_list))
        return smoothed

    def generate(self, row: pd.Series) -> Dict[str, np.ndarray]:
        """Generate all targets for a segment"""
        temporal = self.to_soft_target(
            row.get('rainbow_color_temporal_mode'), 
            self.TEMPORAL_MODES
        )
        spatial = self.to_soft_target(
            row.get('rainbow_color_objectional_mode'),
            self.SPATIAL_MODES
        )
        ontological = self.to_soft_target(
            row.get('rainbow_color_ontological_mode'),
            self.ONTOLOGICAL_MODES
        )

        # Confidence: 0 for Black Album, 1 otherwise
        is_black = all(
            pd.isna(x) or x is None or x == 'None' 
            for x in [
                row.get('rainbow_color_temporal_mode'),
                row.get('rainbow_color_objectional_mode'),
                row.get('rainbow_color_ontological_mode')
            ]
        )
        confidence = np.array([0.0 if is_black else 1.0])

        return {
            'temporal': temporal,
            'spatial': spatial,
            'ontological': ontological,
            'confidence': confidence,
        }


class RainbowRegressionDataset(Dataset):
    """Dataset with runtime soft target generation and pre-computed embeddings"""

    def __init__(self, df: pd.DataFrame, embedding_loader: Optional[PrecomputedEmbeddingLoader] = None, label_smoothing: float = 0.1):
        self.df = df
        self.generator = SoftTargetGenerator(label_smoothing)
        self.embedding_loader = embedding_loader

        # Check if df has inline embeddings
        self._has_inline_embeddings = 'embedding' in df.columns

        if self.embedding_loader:
            print(f"Using pre-computed embeddings from loader")
        elif self._has_inline_embeddings:
            print(f"Using inline embeddings from dataframe")
        else:
            print(f"WARNING: No embeddings available - using random (results meaningless)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        targets = self.generator.generate(row)

        # Load embedding from pre-computed source
        if self.embedding_loader is not None:
            embedding = self.embedding_loader.get_embedding_by_idx(idx)
            embedding = torch.tensor(embedding, dtype=torch.float32)
        elif self._has_inline_embeddings:
            embedding = np.array(row['embedding'], dtype=np.float32)
            embedding = torch.tensor(embedding, dtype=torch.float32)
        else:
            # Fallback to random (should not happen in production)
            embedding = torch.randn(768)

        return {
            'embedding': embedding,
            'temporal': torch.tensor(targets['temporal'], dtype=torch.float32),
            'spatial': torch.tensor(targets['spatial'], dtype=torch.float32),
            'ontological': torch.tensor(targets['ontological'], dtype=torch.float32),
            'confidence': torch.tensor(targets['confidence'], dtype=torch.float32),
        }


class RegressionHead(nn.Module):
    """Rainbow Table regression head"""

    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.3):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.temporal_head = nn.Linear(hidden_dim, 3)
        self.spatial_head = nn.Linear(hidden_dim, 3)
        self.ontological_head = nn.Linear(hidden_dim, 3)
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.shared(x)
        return {
            'temporal': F.softmax(self.temporal_head(x), dim=-1),
            'spatial': F.softmax(self.spatial_head(x), dim=-1),
            'ontological': F.softmax(self.ontological_head(x), dim=-1),
            'confidence': torch.sigmoid(self.confidence_head(x)),
        }


class MultiTaskLoss(nn.Module):
    """Combined regression loss"""

    def __init__(self, weights: Dict[str, float]):
        super().__init__()
        self.weights = weights
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.bce = nn.BCELoss()

    def forward(self, preds, targets):
        losses = {}

        # KL divergence for distributions
        losses['temporal'] = self.kl_div(
            preds['temporal'].log(), targets['temporal']
        )
        losses['spatial'] = self.kl_div(
            preds['spatial'].log(), targets['spatial']
        )
        losses['ontological'] = self.kl_div(
            preds['ontological'].log(), targets['ontological']
        )

        # BCE for confidence
        losses['confidence'] = self.bce(
            preds['confidence'], targets['confidence']
        )

        total = sum(self.weights[k] * v for k, v in losses.items())
        return total, losses


def compute_metrics(preds, targets):
    """Compute regression metrics"""
    metrics = {}

    for dim in ['temporal', 'spatial', 'ontological']:
        pred = preds[dim].cpu().numpy()
        targ = targets[dim].cpu().numpy()

        mae = mean_absolute_error(targ.flatten(), pred.flatten())
        metrics[f'{dim}_mae'] = mae

        # Mode accuracy (argmax matches)
        pred_mode = pred.argmax(axis=1)
        targ_mode = targ.argmax(axis=1)
        accuracy = (pred_mode == targ_mode).mean()
        metrics[f'{dim}_accuracy'] = accuracy

    # Confidence MAE
    conf_pred = preds['confidence'].cpu().numpy().flatten()
    conf_targ = targets['confidence'].cpu().numpy().flatten()
    metrics['confidence_mae'] = mean_absolute_error(conf_targ, conf_pred)

    return metrics


def train_epoch(model, loader, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc='Training'):
        emb = batch['embedding'].to(device)
        targets = {k: v.to(device) for k, v in batch.items() if k != 'embedding'}

        preds = model(emb)
        loss, _ = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = {k: [] for k in ['temporal', 'spatial', 'ontological', 'confidence']}
    all_targets = {k: [] for k in ['temporal', 'spatial', 'ontological', 'confidence']}

    with torch.no_grad():
        for batch in tqdm(loader, desc='Validating'):
            emb = batch['embedding'].to(device)
            targets = {k: v.to(device) for k, v in batch.items() if k != 'embedding'}

            preds = model(emb)
            loss, _ = criterion(preds, targets)

            total_loss += loss.item()

            for k in all_preds.keys():
                all_preds[k].append(preds[k])
                all_targets[k].append(targets[k])

    # Concatenate
    preds_cat = {k: torch.cat(v) for k, v in all_preds.items()}
    targets_cat = {k: torch.cat(v) for k, v in all_targets.items()}

    metrics = compute_metrics(preds_cat, targets_cat)
    metrics['val_loss'] = total_loss / len(loader)

    return metrics


def main():
    """Main training loop"""

    # Load config
    with open('config_regression.yml') as f:
        config = yaml.safe_load(f)

    # Initialize wandb
    wandb.init(
        project=config['logging']['wandb']['project'],
        config=config,
        tags=['phase4', 'regression', 'runpod']
    )

    # Load data
    print("\\n📊 Loading data...")
    df = pd.read_parquet('/workspace/data/base_manifest_db.parquet')
    print(f"   Loaded {len(df)} segments")

    # Load embeddings
    print("\\n🔗 Loading embeddings...")
    data_dir = Path('/workspace/data')
    embedding_file = find_embedding_file(data_dir)
    embedding_loader = None
    if embedding_file:
        embedding_loader = PrecomputedEmbeddingLoader(embedding_file)
    else:
        print("   WARNING: No embedding file found - training with random embeddings!")
        print("   Expected: training_data_with_embeddings.parquet or training_segments_media.parquet")

    # Split
    train_size = int(len(df) * config['data']['train_split'])
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    train_dataset = RainbowRegressionDataset(
        train_df,
        embedding_loader=embedding_loader,
        label_smoothing=config['soft_targets']['label_smoothing']
    )
    val_dataset = RainbowRegressionDataset(
        val_df,
        embedding_loader=embedding_loader,
        label_smoothing=config['soft_targets']['label_smoothing']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )

    print(f"   Train: {len(train_dataset)} segments")
    print(f"   Val: {len(val_dataset)} segments")

    # Initialize model
    device = torch.device('cuda')
    model = RegressionHead(
        input_dim=config['model']['text_encoder']['hidden_size'],
        hidden_dim=config['model']['regression_head']['hidden_dims'][0],
        dropout=config['model']['regression_head']['dropout']
    ).to(device)

    print(f"\\n🏗️  Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Setup training
    criterion = MultiTaskLoss(config['model']['multitask']['loss_weights'])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    # Training loop
    print("\\n🚀 Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config['training']['epochs']):
        print(f"\\nEpoch {epoch+1}/{config['training']['epochs']}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"  Temporal Acc: {val_metrics['temporal_accuracy']:.3f}")
        print(f"  Spatial Acc: {val_metrics['spatial_accuracy']:.3f}")
        print(f"  Ontological Acc: {val_metrics['ontological_accuracy']:.3f}")

        # Log to wandb
        wandb.log({
            'train_loss': train_loss,
            **val_metrics,
            'epoch': epoch
        })

        scheduler.step(val_metrics['val_loss'])

        # Save best
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0
            torch.save(model.state_dict(), '/workspace/output/phase4_best.pt')
            print("  ✅ Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= config['training']['early_stopping']['patience']:
                print(f"\\n⏹️  Early stopping (patience={patience_counter})")
                break

    print("\\n✅ Training complete!")
    print(f"   Best val loss: {best_val_loss:.4f}")

    wandb.finish()


if __name__ == '__main__':
    main()
'''

    trainer_path = training_dir / "train_phase4.py"
    trainer_path.write_text(trainer_code)
    trainer_path.chmod(0o755)

    print(f"✅ Created {trainer_path}")
    print()


def run_training(training_dir: Path):
    """Run Phase 4 training"""
    print("=" * 80)
    print("STARTING PHASE 4 TRAINING")
    print("=" * 80)
    print()

    import os

    os.chdir(training_dir)

    # Run training script
    subprocess.check_call([sys.executable, "train_phase4.py"])


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase 4 RunPod Launcher")
    parser.add_argument(
        "--skip-deps", action="store_true", help="Skip dependency installation"
    )
    args = parser.parse_args()

    print("\\n" + "=" * 80)
    print("PHASE 4 REGRESSION TRAINING - RUNPOD")
    print("Rainbow Table Ontological Mode Prediction")
    print("=" * 80 + "\\n")

    if not args.skip_deps:
        install_dependencies()

    check_gpu()
    check_data()
    training_dir = setup_training_code()
    create_phase4_trainer(training_dir)
    run_training(training_dir)

    print("\\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("Output: /workspace/output/phase4_best.pt")
    print("=" * 80 + "\\n")


if __name__ == "__main__":
    main()
