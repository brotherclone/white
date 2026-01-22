#!/usr/bin/env python3
"""
RunPod Training Launch Script for Rainbow Pipeline.

Usage:
    python runpod_train.py --mode multiclass
    python runpod_train.py --mode binary
    python runpod_train.py --mode multilabel
"""

import subprocess
import sys
from pathlib import Path


def install_dependencies():
    """Install required packages."""
    print("=" * 60)
    print("INSTALLING DEPENDENCIES")
    print("=" * 60)

    packages = [
        "torch",
        "transformers",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "pyyaml",
        "tqdm",
        "wandb",
        "matplotlib",
        "seaborn",
    ]

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])
    print("Dependencies installed.\n")


def check_data():
    """Verify training data exists."""
    print("=" * 60)
    print("CHECKING DATA")
    print("=" * 60)

    data_dir = Path("/workspace/data")
    required_files = [
        "base_manifest_db.parquet",
        "training_data_full.parquet",
    ]

    for fname in required_files:
        fpath = data_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"  [OK] {fname} ({size_mb:.1f} MB)")
        else:
            print(f"  [MISSING] {fname}")
            raise FileNotFoundError(f"Required file not found: {fpath}")

    # Check for the big embedded file (optional but nice to have)
    embedded = data_dir / "training_data_embedded.parquet"
    if embedded.exists():
        size_gb = embedded.stat().st_size / (1024**3)
        print(f"  [OK] training_data_embedded.parquet ({size_gb:.1f} GB)")

    print()


def clone_repo():
    """Clone the training code if not present."""
    print("=" * 60)
    print("SETTING UP CODE")
    print("=" * 60)

    code_dir = Path("/workspace/white")

    if not code_dir.exists():
        print("Cloning repository...")
        subprocess.check_call(
            ["git", "clone", "https://github.com/brotherclone/white.git", str(code_dir)]
        )
        # Checkout the feature branch with multiclass support
        subprocess.check_call(
            ["git", "-C", str(code_dir), "checkout", "feature/dataPrep"]
        )
    else:
        print("Repository already exists, pulling latest...")
        subprocess.check_call(["git", "-C", str(code_dir), "pull"])

    print()
    return code_dir


def run_training(mode: str = "multiclass"):
    """Run the training script."""
    print("=" * 60)
    print(f"STARTING {mode.upper()} TRAINING")
    print("=" * 60)

    import os

    os.chdir("/workspace/white/training")

    # Select config based on mode
    config_map = {
        "binary": "config.yml",
        "multiclass": "config_multiclass.yml",
        "multilabel": "config_multilabel.yml",
    }

    config_file = config_map.get(mode, "config_multiclass.yml")

    # Update config to use RunPod paths
    update_config_paths(config_file)

    # Import and run
    import yaml
    import torch

    print(f"Config: {config_file}")
    print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()

    # Run training
    from train import Trainer

    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Set seed
    seed = config["reproducibility"]["seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    trainer = Trainer(config)
    trainer.train()


def update_config_paths(config_file: str):
    """Update config file paths for RunPod environment."""
    import yaml

    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Update data path
    config["data"]["manifest_path"] = "/workspace/data/base_manifest_db.parquet"

    # Update output directory
    config["logging"]["output_dir"] = "/workspace/output"

    # Ensure CUDA device
    config["training"]["device"] = "cuda"
    config["training"]["mixed_precision"] = True

    # Save updated config
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Updated {config_file} for RunPod paths")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RunPod Training Launcher")
    parser.add_argument(
        "--mode",
        choices=["binary", "multiclass", "multilabel"],
        default="multiclass",
        help="Training mode (default: multiclass)",
    )
    parser.add_argument(
        "--skip-deps", action="store_true", help="Skip dependency installation"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("RAINBOW PIPELINE - RUNPOD TRAINING")
    print("=" * 60 + "\n")

    if not args.skip_deps:
        install_dependencies()

    check_data()
    clone_repo()
    run_training(args.mode)


if __name__ == "__main__":
    main()
