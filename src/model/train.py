import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CHECKPOINT_DIR, DATA_DIR
from src.data.dataset import SignAvatarDataset
from src.model.architecture import SignMotionGenerator, count_parameters
from src.model.loss import SignMotionLoss, WeightedSignMotionLoss


def select_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def load_vocab_size() -> int:
    vocab_path = DATA_DIR / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"vocab.json not found at {vocab_path}. Run prepare_data first.")
    with vocab_path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)
    return len(vocab)


def load_checkpoint(checkpoint_path: Path) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    config: dict,
    vocab_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config,
        "vocab_size": vocab_size,
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict:
    model.train()
    totals: dict[str, float] = {}
    num_batches = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        gloss_id = batch["gloss_id"].to(device)
        motion = batch["motion"].to(device)

        optimizer.zero_grad()

        # Forward pass
        pred_motion = model(gloss_id)

        # Loss
        loss_dict = loss_fn(pred_motion, motion)
        loss = loss_dict["loss"]

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate metrics
        for key, value in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + value.item()
        num_batches += 1

    return {key: value / num_batches for key, value in totals.items()}


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    totals: dict[str, float] = {}
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            gloss_id = batch["gloss_id"].to(device)
            motion = batch["motion"].to(device)

            pred_motion = model(gloss_id)
            loss_dict = loss_fn(pred_motion, motion)

            for key, value in loss_dict.items():
                totals[key] = totals.get(key, 0.0) + value.item()
            num_batches += 1

    if num_batches == 0:
        return {"loss": float("inf")}

    return {key: value / num_batches for key, value in totals.items()}


def create_loss(loss_type: str) -> nn.Module:
    if loss_type == "standard":
        return SignMotionLoss()
    if loss_type == "weighted":
        return WeightedSignMotionLoss()
    raise ValueError("loss_type must be one of: standard, weighted")


def train_model(
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    seq_len: int = 60,
    motion_dim: int = 182,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 4,
    dropout: float = 0.1,
    num_workers: int = 0,
    save_every: int = 10,
    resume_from: Optional[str] = None,
    loss_type: str = "standard",
    sequence_mode: str = "pad_trim",
) -> list[dict]:
    if loss_type not in {"standard", "weighted"}:
        raise ValueError("loss_type must be one of: standard, weighted")
    if sequence_mode not in {"pad_trim", "resample"}:
        raise ValueError("sequence_mode must be one of: pad_trim, resample")

    device = select_device()

    # Load vocab size
    vocab_size = load_vocab_size()
    print(f"Vocab size: {vocab_size}")

    # Load datasets
    print("Loading datasets...")
    train_ds = SignAvatarDataset(split="train", seq_len=seq_len, sequence_mode=sequence_mode)
    val_ds = SignAvatarDataset(split="val", seq_len=seq_len, sequence_mode=sequence_mode)

    if len(train_ds) == 0:
        raise RuntimeError("Training dataset is empty. Run prepare_data first.")
    if len(val_ds) == 0:
        print("Warning: Validation dataset is empty. Using training loss for best model selection.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Create model
    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seq_len": seq_len,
        "motion_dim": motion_dim,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dropout": dropout,
        "loss_type": loss_type,
        "sequence_mode": sequence_mode,
    }
    model = SignMotionGenerator(
        vocab_size=vocab_size,
        seq_len=seq_len,
        motion_dim=motion_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    print(f"Model parameters: {count_parameters(model)}")

    # Loss, optimizer, scheduler
    loss_fn = create_loss(loss_type).to(device)
    print(f"Loss type: {loss_type}")
    print(f"Sequence mode: {sequence_mode}")
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Resume from checkpoint
    start_epoch = 0
    if resume_from:
        checkpoint = load_checkpoint(Path(resume_from))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    history = []
    best_val_loss = float("inf")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Training
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device)

        # Validation
        val_metrics = validate(model, val_loader, loss_fn, device)

        # Learning rate scheduling
        scheduler.step(val_metrics["loss"] if len(val_ds) > 0 else train_metrics["loss"])

        # History
        current_lr = optimizer.param_groups[0]["lr"]
        history_entry = {
            "epoch": epoch,
            "learning_rate": current_lr,
        }
        for key, value in train_metrics.items():
            history_entry[f"train_{key}"] = value
        for key, value in val_metrics.items():
            history_entry[f"val_{key}"] = value
        history.append(history_entry)

        # Console output
        print(f"Train Loss: {train_metrics['loss']:.6f}")
        if len(val_ds) > 0:
            print(f"Val Loss: {val_metrics['loss']:.6f}")
        for key in sorted(train_metrics):
            if key != "loss":
                print(f"  train_{key}: {train_metrics[key]:.6f}")
        if len(val_ds) > 0:
            for key in sorted(val_metrics):
                if key != "loss":
                    print(f"  val_{key}: {val_metrics[key]:.6f}")

        # Best checkpoint
        eval_loss = val_metrics["loss"] if len(val_ds) > 0 else train_metrics["loss"]
        if eval_loss < best_val_loss:
            best_val_loss = eval_loss
            save_checkpoint(
                CHECKPOINT_DIR / "best_model.pth",
                model,
                optimizer,
                epoch,
                train_metrics["loss"],
                val_metrics["loss"],
                config,
                vocab_size,
            )
            print(f"Best Val Loss: {best_val_loss:.6f}")

        # Periodic checkpoint
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                CHECKPOINT_DIR / f"epoch_{epoch:03d}.pth",
                model,
                optimizer,
                epoch,
                train_metrics["loss"],
                val_metrics["loss"],
                config,
                vocab_size,
            )

    # Final checkpoint
    save_checkpoint(
        CHECKPOINT_DIR / "final_model.pth",
        model,
        optimizer,
        epochs - 1,
        train_metrics["loss"],
        val_metrics["loss"],
        config,
        vocab_size,
    )

    # Save history
    logs_dir = Path("outputs/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    history_path = logs_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved training history to {history_path}")

    return history


if __name__ == "__main__":
    # Quick smoke test: 1 epoch, small batch
    print("Running smoke test: 1 epoch, batch_size=4")
    history = train_model(epochs=1, batch_size=4, save_every=1)
    print("Smoke test completed successfully.")
