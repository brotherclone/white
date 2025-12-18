"""
try script to verify everything works before full training.

Usage:
    python try_setup.py --manifest ../data/base_manifest_db.parquet
"""

import argparse
import torch
from transformers import AutoTokenizer
from pathlib import Path

# Local imports
from core.pipeline import build_dataloaders, load_manifest
from models.text_encoder import TextEncoder
from models.classifier import BinaryClassifier, RainbowModel


def try_data_loading(manifest_path: str):
    """try data loading and filtering."""
    print("\n" + "=" * 60)
    print("try 1: Data Loading")
    print("=" * 60)

    try:
        df = load_manifest(
            manifest_path=manifest_path,
            require_concept=True,
            min_concept_length=50,
        )
        print(f"✓ Successfully loaded {len(df)} tracks")

        # Check columns
        required_cols = ["id", "concept", "training_data"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"✗ Missing columns: {missing}")
            return False
        print("✓ All required columns present")

        # Check training_data structure
        sample = df.iloc[0]["training_data"]
        if not isinstance(sample, dict):
            print(f"✗ training_data is not dict: {type(sample)}")
            return False
        if "has_rebracketing_markers" not in sample:
            print("✗ has_rebracketing_markers not in training_data")
            return False
        print("✓ training_data structure valid")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def try_dataloader(manifest_path: str):
    """try dataloader creation and batching."""
    print("\n" + "=" * 60)
    print("try 2: DataLoader")
    print("=" * 60)

    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

        train_loader, val_loader, stats = build_dataloaders(
            manifest_path=manifest_path,
            tokenizer=tokenizer,
            target_column="has_rebracketing_markers",
            batch_size=4,
            num_workers=0,  # Single process for trying
        )

        print("✓ Created dataloaders")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")

        # try batch
        batch = next(iter(train_loader))
        print("✓ Fetched batch")
        print(f"  Keys: {list(batch.keys())}")
        print(f"  input_ids: {batch['input_ids'].shape}")
        print(f"  labels: {batch['labels'].shape}")
        print(f"  Sample labels: {batch['labels'].tolist()}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def try_model():
    """try model initialization and forward pass."""
    print("\n" + "=" * 60)
    print("try 3: Model")
    print("=" * 60)

    try:
        # Create model
        text_encoder = TextEncoder(
            model_name="microsoft/deberta-v3-base",
            hidden_size=768,
            freeze_layers=0,
            pooling="mean",
        )
        print("✓ Created text encoder")

        classifier = BinaryClassifier(
            input_dim=768,
            hidden_dims=[256, 128],
            dropout=0.3,
        )
        print("✓ Created classifier")

        model = RainbowModel(text_encoder, classifier)
        print("✓ Created combined model")

        # try forward pass
        batch_size = 4
        seq_len = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        logits = model(input_ids, attention_mask)
        print("✓ Forward pass successful")
        print(f"  Input: {input_ids.shape}")
        print(f"  Output: {logits.shape}")

        # Check output shape
        if logits.shape != (batch_size,):
            print(f"✗ Wrong output shape: expected ({batch_size},), got {logits.shape}")
            return False

        print("✓ Output shape correct")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def try_training_step(manifest_path: str):
    """try a single training step."""
    print("\n" + "=" * 60)
    print("try 4: Training Step")
    print("=" * 60)

    try:
        # Setup
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

        train_loader, val_loader, stats = build_dataloaders(
            manifest_path=manifest_path,
            tokenizer=tokenizer,
            target_column="has_rebracketing_markers",
            batch_size=2,
            num_workers=0,
        )

        model = RainbowModel(
            TextEncoder("microsoft/deberta-v3-base", pooling="mean"),
            BinaryClassifier(768, [256], dropout=0.3),
        )

        # Get batch
        batch = next(iter(train_loader))
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        print("✓ Prepared batch")

        # Forward
        logits = model(input_ids, attention_mask)
        print("✓ Forward pass")

        # Loss
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(logits, labels)
        print(f"✓ Loss computation: {loss.item():.4f}")

        # Backward
        loss.backward()
        print("✓ Backward pass")

        # Check gradients
        has_grad = any(p.grad is not None for p in model.parameters())
        if not has_grad:
            print("✗ No gradients computed")
            return False
        print("✓ Gradients computed")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="try Rainbow Pipeline setup")
    parser.add_argument(
        "--manifest",
        type=str,
        default="../data/base_manifest_db.parquet",
        help="Path to manifest file",
    )
    args = parser.parse_args()

    # Check manifest exists
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"✗ Manifest not found: {manifest_path}")
        print("  Please provide correct path with --manifest")
        return

    # Run trys
    trys = [
        ("Data Loading", lambda: try_data_loading(str(manifest_path))),
        ("DataLoader", lambda: try_dataloader(str(manifest_path))),
        ("Model", try_model),
        ("Training Step", lambda: try_training_step(str(manifest_path))),
    ]

    results = []
    for name, try_fn in trys:
        try:
            result = try_fn()
            results.append((name, result))
        except Exception as e:
            print(f"✗ try '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("try SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} {name}")

    all_passed = all(r[1] for r in results)
    print()
    if all_passed:
        print("🎉 All trys passed! Ready to train.")
    else:
        print("❌ Some trys failed. Fix issues before training.")


if __name__ == "__main__":
    main()
