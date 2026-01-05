"""
Main training script for Rainbow Pipeline.

Usage:
    python train.py --config config.yml
"""

import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import json

# Local imports
from core.pipeline import build_dataloaders
from models.text_encoder import TextEncoder
from models.classifier import BinaryClassifier, RainbowModel


class Trainer:
    """Training loop manager."""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config["training"]["device"])

        # Setup output directory
        self.output_dir = Path(config["logging"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.output_dir / "config.yml", "w") as f:
            yaml.dump(config, f)

        # Setup model
        self.setup_model()

        # Setup data
        self.setup_data()

        # Setup training
        self.setup_training()

        # Tracking
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def setup_model(self):
        """Initialize model components."""
        print("\n" + "=" * 60)
        print("SETTING UP MODEL")
        print("=" * 60)

        # Text encoder
        text_config = self.config["model"]["text_encoder"]
        self.text_encoder = TextEncoder(
            model_name=text_config["model_name"],
            hidden_size=text_config["hidden_size"],
            freeze_layers=text_config["freeze_layers"],
            pooling=text_config["pooling"],
        )

        # Classifier
        clf_config = self.config["model"]["classifier"]
        self.classifier = BinaryClassifier(
            input_dim=self.text_encoder.hidden_size,
            hidden_dims=clf_config["hidden_dims"],
            dropout=clf_config["dropout"],
            activation=clf_config["activation"],
        )

        # Combined model
        self.model = RainbowModel(
            text_encoder=self.text_encoder,
            classifier=self.classifier,
        )
        self.model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def setup_data(self):
        """Initialize dataloaders."""
        print("\n" + "=" * 60)
        print("SETTING UP DATA")
        print("=" * 60)

        # Resolve model name (preferred location: model.text_encoder.model_name)
        model_name = self.config.get("model_name_or_path") or self.config.get(
            "model", {}
        ).get("text_encoder", {}).get("model_name")
        if not model_name:
            raise KeyError(
                "model name not found in config; expected `model.text_encoder.model_name` "
                "or top-level `model_name_or_path`."
            )

        # Tokenizer
        self.tokenizer = self.load_tokenizer(
            model_name,
            cache_dir=self.config.get("cache_dir", None),
            local_files_only=self.config.get("local_files_only", False),
            trust_remote_code=self.config.get("trust_remote_code", False),
        )

        # Dataloaders
        data_config = self.config["data"]
        train_config = self.config["training"]

        self.train_loader, self.val_loader, self.stats = build_dataloaders(
            manifest_path=data_config["manifest_path"],
            tokenizer=self.tokenizer,
            target_column=data_config["target_column"],
            train_split=data_config["train_split"],
            val_split=data_config["val_split"],
            batch_size=train_config["batch_size"],
            num_workers=train_config["num_workers"],
            random_seed=data_config["random_seed"],
            require_concept=data_config["require_concept"],
            min_concept_length=data_config["min_concept_length"],
            max_length=self.config["model"]["text_encoder"]["max_length"],
        )

        # Use pos_weight for imbalanced classes
        self.pos_weight = torch.tensor([self.stats["pos_weight"]]).to(self.device)
        print(f"\nUsing pos_weight: {self.pos_weight.item():.3f}")

    def setup_training(self):
        """Initialize optimizer, scheduler, loss."""
        train_config = self.config["training"]

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_config["learning_rate"],
            weight_decay=train_config["weight_decay"],
        )

        # Scheduler with warmup
        num_training_steps = len(self.train_loader) * train_config["epochs"]
        num_warmup_steps = int(num_training_steps * train_config["warmup_ratio"])

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=num_warmup_steps,
        )

        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps - num_warmup_steps,
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[num_warmup_steps],
        )

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        # Mixed precision
        self.use_amp = train_config["mixed_precision"] and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        print("\nTraining setup:")
        print(f"  Optimizer: AdamW (lr={train_config['learning_rate']})")
        print("  Scheduler: Warmup + Cosine")
        print(f"  Mixed precision: {self.use_amp}")

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

            # Backward
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["training"]["max_grad_norm"]
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["training"]["max_grad_norm"]
                )
                self.optimizer.step()

            self.scheduler.step()

            # Metrics
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            if batch_idx % self.config["logging"]["log_every_n_steps"] == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{100 * correct / total:.2f}%",
                    }
                )

        return {
            "loss": total_loss / len(self.train_loader),
            "accuracy": correct / total,
        }

    @torch.no_grad()
    def validate(self) -> dict:
        """Validate on validation set."""
        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(self.val_loader, desc="Validating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return {
            "loss": total_loss / len(self.val_loader),
            "accuracy": correct / total,
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        # Save latest
        torch.save(checkpoint, self.output_dir / "checkpoint_latest.pt")

        # Save best
        if is_best:
            torch.save(checkpoint, self.output_dir / "checkpoint_best.pt")
            print(f"  ✓ Saved best model (val_loss: {self.best_val_loss:.4f})")

    def train(self):
        """Main training loop."""
        print("\n" + "=" * 60)
        print("TRAINING")
        print("=" * 60)

        num_epochs = self.config["training"]["epochs"]
        early_stopping_config = self.config["training"]["early_stopping"]

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)

            # Train
            train_metrics = self.train_epoch(epoch)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])

            # Validate
            val_metrics = self.validate()
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])

            # Print metrics
            print(
                f"\nTrain Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}"
            )
            print(
                f"Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.4f}"
            )

            # Check for improvement
            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

            # Early stopping
            if early_stopping_config["enabled"]:
                if self.epochs_without_improvement >= early_stopping_config["patience"]:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break

        # Save training history
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {self.output_dir}")

    def load_tokenizer(self, pretrained_name_or_path: str, **kwargs) -> AutoTokenizer:
        """
        Load HF tokenizer with robust three-tier fallback system.

        Tier 1: Fast tokenizer (preferred)
        Tier 2: Slow tokenizer with add_prefix_space=False (DeBERTa-v3 fix)
        Tier 3: Direct slow tokenizer class loading (bypass AutoTokenizer)
        """
        want_fast = kwargs.pop("use_fast", True)

        # TIER 1: Try fast tokenizer
        if want_fast:
            try:
                return AutoTokenizer.from_pretrained(
                    pretrained_name_or_path, use_fast=True, **kwargs
                )
            except (AttributeError, Exception) as e:
                msg = str(e)
                if "has no attribute 'endswith'" in msg or "NoneType" in msg:
                    print(
                        f"⚠️  Fast tokenizer failed ({type(e).__name__}), falling back to slow tokenizer..."
                    )
                else:
                    # Re-raise if it's a different error we should know about
                    raise

        # TIER 2: Slow tokenizer with DeBERTa-v3 fix
        # Make sure use_fast isn't lingering in kwargs
        kwargs_tier2 = kwargs.copy()
        kwargs_tier2.pop("use_fast", None)

        try:
            return AutoTokenizer.from_pretrained(
                pretrained_name_or_path,
                use_fast=False,
                add_prefix_space=False,  # Critical for DeBERTa-v3
                **kwargs_tier2,
            )
        except Exception as e:
            print(
                f"⚠️  Slow tokenizer failed ({type(e).__name__}), bypassing AutoTokenizer..."
            )

        # TIER 3: NUCLEAR OPTION - Load slow tokenizer class directly
        # AutoTokenizer is broken for this model, so we bypass it entirely
        print(f"🔥 Loading tokenizer class directly for {pretrained_name_or_path}")

        from transformers import DebertaV2Tokenizer

        # Clean kwargs completely
        kwargs_tier3 = {
            "cache_dir": kwargs.get("cache_dir"),
            "local_files_only": kwargs.get("local_files_only", False),
        }
        # Remove None values
        kwargs_tier3 = {k: v for k, v in kwargs_tier3.items() if v is not None}

        return DebertaV2Tokenizer.from_pretrained(
            pretrained_name_or_path, add_prefix_space=False, **kwargs_tier3
        )


def main():
    parser = argparse.ArgumentParser(description="Train Rainbow Pipeline")
    parser.add_argument(
        "--config", type=str, default="config.yml", help="Path to config file"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Set random seeds
    seed = config["reproducibility"]["seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Train
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
