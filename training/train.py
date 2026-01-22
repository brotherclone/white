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
from core.multiclass_pipeline import build_multiclass_dataloaders
from core.multiclass_metrics import MultiClassMetrics, MultiLabelMetrics, top_k_accuracy
from models.text_encoder import TextEncoder
from models.classifier import BinaryClassifier, RainbowModel
from models.multiclass_classifier import (
    MultiClassRebracketingClassifier,
    MultiClassRainbowModel,
)

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Trainer:
    """Training loop manager."""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config["training"]["device"])

        # Mode detection: binary, multiclass, or multilabel
        self.target_type = config["data"].get("target_type", "binary")
        self.is_binary = self.target_type == "binary"
        self.is_multiclass = self.target_type == "multiclass"
        self.is_multilabel = self.target_type == "multilabel"

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
        self.best_val_metric = 0.0 if not self.is_binary else float("inf")
        self.epochs_without_improvement = 0
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        if not self.is_binary:
            self.history["val_macro_f1"] = []
            self.history["val_micro_f1"] = []

    def setup_model(self):
        """Initialize model components."""
        print("\n" + "=" * 60)
        print("SETTING UP MODEL")
        print("=" * 60)
        print(f"Mode: {self.target_type}")

        # Text encoder (shared between binary and multiclass)
        text_config = self.config["model"]["text_encoder"]
        self.text_encoder = TextEncoder(
            model_name=text_config["model_name"],
            hidden_size=text_config["hidden_size"],
            freeze_layers=text_config["freeze_layers"],
            pooling=text_config["pooling"],
        )

        clf_config = self.config["model"]["classifier"]

        if self.is_binary:
            # Binary classifier
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

            # Initialize multiclass-specific attributes to None
            self.num_classes = None
            self.multi_label = False
            self.class_names = None
            self.class_mapping = None
        else:
            # Multiclass/Multilabel classifier
            data_config = self.config["data"]
            self.class_mapping = data_config["class_mapping"]
            self.num_classes = len(self.class_mapping)
            self.multi_label = self.is_multilabel
            self.class_names = list(self.class_mapping.keys())

            self.classifier = MultiClassRebracketingClassifier(
                input_dim=self.text_encoder.hidden_size,
                num_classes=self.num_classes,
                hidden_dims=clf_config["hidden_dims"],
                dropout=clf_config["dropout"],
                activation=clf_config["activation"],
                multi_label=self.multi_label,
            )

            # Combined model
            self.model = MultiClassRainbowModel(
                text_encoder=self.text_encoder,
                classifier=self.classifier,
            )

            print(f"Number of classes: {self.num_classes}")
            print(f"Multi-label: {self.multi_label}")
            print(f"Classes: {self.class_names}")

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

        if self.is_binary:
            # Binary mode: use original pipeline
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
            self.class_weights = None
            self.label_encoder = None
            print(f"\nUsing pos_weight: {self.pos_weight.item():.3f}")
        else:
            # Multiclass/Multilabel mode: use multiclass pipeline
            self.train_loader, self.val_loader, self.stats = (
                build_multiclass_dataloaders(
                    manifest_path=data_config["manifest_path"],
                    tokenizer=self.tokenizer,
                    class_mapping=self.class_mapping,
                    target_column=data_config["target_column"],
                    train_split=data_config["train_split"],
                    val_split=data_config["val_split"],
                    batch_size=train_config["batch_size"],
                    num_workers=train_config["num_workers"],
                    random_seed=data_config["random_seed"],
                    require_concept=data_config["require_concept"],
                    min_concept_length=data_config["min_concept_length"],
                    max_length=self.config["model"]["text_encoder"]["max_length"],
                    multi_label=self.multi_label,
                    stratified=data_config.get("stratified", True),
                    filter_unknown_types=data_config.get("filter_unknown_types", True),
                )
            )

            # Store class weights and label encoder for multiclass
            self.class_weights = torch.tensor(
                self.stats["class_weights"], dtype=torch.float32
            ).to(self.device)
            self.label_encoder = self.stats["label_encoder"]
            self.pos_weight = None
            print(f"\nClass weights loaded for {self.num_classes} classes")

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

        # Loss function - differs by mode
        if self.is_binary:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            self.metrics_tracker = None
            print("  Loss: BCEWithLogitsLoss (binary)")
        elif self.is_multilabel:
            # Multi-label: BCEWithLogitsLoss with class weights as pos_weight
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
            self.metrics_tracker = MultiLabelMetrics(
                num_classes=self.num_classes,
                class_names=self.class_names,
            )
            print("  Loss: BCEWithLogitsLoss (multi-label)")
        else:
            # Single-label multiclass: CrossEntropyLoss with class weights
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            self.metrics_tracker = MultiClassMetrics(
                num_classes=self.num_classes,
                class_names=self.class_names,
            )
            print("  Loss: CrossEntropyLoss (multiclass)")

        # Mixed precision
        self.use_amp = train_config["mixed_precision"] and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        print("\nTraining setup:")
        print(f"  Optimizer: AdamW (lr={train_config['learning_rate']})")
        print("  Scheduler: Warmup + Cosine")
        print(f"  Mixed precision: {self.use_amp}")

        # Setup WandB if enabled
        self.setup_wandb()

    def setup_wandb(self):
        """Initialize WandB logging if enabled in config."""
        self.use_wandb = False

        # Check if WandB is enabled in config
        logging_config = self.config.get("logging", {})
        wandb_config = logging_config.get("wandb", {})

        if not wandb_config.get("enabled", False):
            return

        if not WANDB_AVAILABLE:
            print("Warning: WandB enabled in config but not installed. Skipping.")
            return

        # Initialize WandB
        project_name = wandb_config.get("project", "rainbow-pipeline")
        run_name = wandb_config.get("run_name", None)

        wandb.init(
            project=project_name,
            name=run_name,
            config=self.config,
            dir=str(self.output_dir),
        )

        self.use_wandb = True
        print(f"  WandB: Logging to project '{project_name}'")

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

            # Compute accuracy based on mode
            if self.is_binary:
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            elif self.is_multilabel:
                preds = (torch.sigmoid(logits) > 0.5).float()
                # Per-label accuracy for multi-label
                correct += (preds == labels).sum().item()
                total += labels.numel()
            else:
                # Multiclass: argmax
                preds = torch.argmax(logits, dim=-1)
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

                # Log to WandB per step
                if self.use_wandb:
                    global_step = epoch * len(self.train_loader) + batch_idx
                    wandb.log(
                        {
                            "train/step_loss": loss.item(),
                            "train/step_acc": correct / total,
                            "global_step": global_step,
                        },
                        step=global_step,
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
        all_logits = []
        all_labels = []

        # Reset metrics tracker if using multiclass
        if self.metrics_tracker is not None:
            self.metrics_tracker.reset()

        for batch in tqdm(self.val_loader, desc="Validating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()

            # Compute accuracy based on mode
            if self.is_binary:
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            elif self.is_multilabel:
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.numel()
                # Update metrics tracker
                self.metrics_tracker.update(logits, labels)
            else:
                # Multiclass: argmax
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                # Update metrics tracker
                self.metrics_tracker.update(logits, labels)
                # Store for top-k accuracy
                all_logits.append(logits)
                all_labels.append(labels)

        # Build result dict
        result = {
            "loss": total_loss / len(self.val_loader),
            "accuracy": correct / total,
        }

        # Add comprehensive multiclass metrics
        if self.metrics_tracker is not None:
            tracker_metrics = self.metrics_tracker.compute()
            result.update(tracker_metrics)

            # Add top-k accuracy for single-label multiclass
            if not self.is_multilabel and all_logits:
                all_logits = torch.cat(all_logits, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                for k in [3, 5]:
                    if k < self.num_classes:
                        result[f"top_{k}_accuracy"] = top_k_accuracy(
                            all_logits, all_labels, k=k
                        )

        return result

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "target_type": self.target_type,
        }

        # Add multiclass-specific metadata
        if not self.is_binary:
            checkpoint["class_mapping"] = self.class_mapping
            checkpoint["num_classes"] = self.num_classes
            checkpoint["multi_label"] = self.multi_label
            checkpoint["best_val_metric"] = self.best_val_metric

        # Save latest
        torch.save(checkpoint, self.output_dir / "checkpoint_latest.pt")

        # Save best
        if is_best:
            torch.save(checkpoint, self.output_dir / "checkpoint_best.pt")
            if self.is_binary:
                print(f"  ✓ Saved best model (val_loss: {self.best_val_loss:.4f})")
            else:
                print(
                    f"  ✓ Saved best model (val_macro_f1: {self.best_val_metric:.4f})"
                )

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

            # Store multiclass metrics in history
            if not self.is_binary:
                self.history["val_macro_f1"].append(val_metrics.get("macro_f1", 0))
                self.history["val_micro_f1"].append(val_metrics.get("micro_f1", 0))

            # Print metrics
            print(
                f"\nTrain Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}"
            )
            print(
                f"Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.4f}"
            )

            # Print additional multiclass metrics
            if not self.is_binary:
                macro_f1 = val_metrics.get("macro_f1", 0)
                micro_f1 = val_metrics.get("micro_f1", 0)
                print(f"Val Macro F1: {macro_f1:.4f} | Val Micro F1: {micro_f1:.4f}")

            # Check for improvement
            # Binary: minimize loss, Multiclass: maximize macro_f1
            if self.is_binary:
                is_best = val_metrics["loss"] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics["loss"]
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
            else:
                current_metric = val_metrics.get("macro_f1", 0)
                is_best = current_metric > self.best_val_metric
                if is_best:
                    self.best_val_metric = current_metric
                    self.best_val_loss = val_metrics["loss"]  # Track loss too
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

            # Log to WandB
            if self.use_wandb:
                log_dict = {
                    "epoch": epoch + 1,
                    "train/loss": train_metrics["loss"],
                    "train/accuracy": train_metrics["accuracy"],
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                }

                # Add multiclass-specific metrics
                if not self.is_binary:
                    log_dict["val/macro_f1"] = val_metrics.get("macro_f1", 0)
                    log_dict["val/micro_f1"] = val_metrics.get("micro_f1", 0)
                    log_dict["val/weighted_f1"] = val_metrics.get("weighted_f1", 0)

                    # Per-class F1 scores
                    for class_name in self.class_names:
                        key = f"f1_{class_name}"
                        if key in val_metrics:
                            log_dict[f"val/f1/{class_name}"] = val_metrics[key]

                wandb.log(log_dict)

            # Early stopping
            if early_stopping_config["enabled"]:
                if self.epochs_without_improvement >= early_stopping_config["patience"]:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break

        # Save training history
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        # Save final artifacts for multiclass
        if not self.is_binary:
            self._save_confusion_matrix()
            self._log_final_wandb_artifacts()

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        if self.is_binary:
            print(f"Best validation loss: {self.best_val_loss:.4f}")
        else:
            print(f"Best validation macro F1: {self.best_val_metric:.4f}")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {self.output_dir}")

    def _save_confusion_matrix(self):
        """Save confusion matrix plot for multiclass models."""
        if self.metrics_tracker is None or self.is_multilabel:
            return

        try:
            save_path = self.output_dir / "confusion_matrix.png"
            fig = self.metrics_tracker.plot_confusion_matrix(
                normalize="true", save_path=str(save_path)
            )

            # Log to WandB
            if self.use_wandb:
                wandb.log({"confusion_matrix": wandb.Image(fig)})

            # Also save classification report
            report = self.metrics_tracker.get_classification_report()
            report_path = self.output_dir / "classification_report.txt"
            with open(report_path, "w") as f:
                f.write(report)
            print(f"  ✓ Saved classification report to {report_path}")

            # Close figure to free memory
            import matplotlib.pyplot as plt

            plt.close(fig)

        except Exception as e:
            print(f"Warning: Could not save confusion matrix: {e}")

    def _log_final_wandb_artifacts(self):
        """Log final artifacts to WandB."""
        if not self.use_wandb:
            return

        try:
            # Log model artifact
            artifact = wandb.Artifact(
                name=f"model-{wandb.run.id}",
                type="model",
                description="Trained multiclass Rainbow model",
            )
            artifact.add_file(str(self.output_dir / "checkpoint_best.pt"))
            wandb.log_artifact(artifact)

            # Log classification report as text
            if self.metrics_tracker is not None and not self.is_multilabel:
                report = self.metrics_tracker.get_classification_report()
                wandb.log({"classification_report": wandb.Html(f"<pre>{report}</pre>")})

                # Log most confused pairs
                confused_pairs = self.metrics_tracker.get_most_confused_pairs(top_k=5)
                if confused_pairs:
                    table = wandb.Table(
                        columns=["True Class", "Predicted Class", "Count"]
                    )
                    for true_cls, pred_cls, count in confused_pairs:
                        table.add_data(true_cls, pred_cls, count)
                    wandb.log({"confused_pairs": table})

            wandb.finish()
            print("  ✓ WandB artifacts logged")

        except Exception as e:
            print(f"Warning: Could not log WandB artifacts: {e}")

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
