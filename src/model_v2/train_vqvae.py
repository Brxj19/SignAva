import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import SignAvatarDataset, denormalize_sequence
from src.model_v2.losses import WeightedMotionReconstructionLoss
from src.model_v2.utils import CHECKPOINT_V2_DIR, OUTPUT_V2_DIR, count_parameters, save_json, select_device
from src.model_v2.vqvae import MotionVQVAE


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"loss": float("inf")}
    keys = rows[0].keys()
    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in keys}


def _run_epoch(model, loader, loss_fn, device, optimizer=None) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    rows: list[dict[str, float]] = []
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="Training" if is_train else "Evaluating", leave=False):
            motion = batch["motion"].to(device)
            if is_train:
                optimizer.zero_grad()
            recon, _, vq_loss, perplexity = model(motion)
            metrics = loss_fn(recon, motion, vq_loss=vq_loss, perplexity=perplexity)
            if is_train:
                metrics["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            rows.append({key: float(value.detach().cpu()) for key, value in metrics.items()})
    return _mean_metrics(rows)


def _save_checkpoint(path: Path, model, optimizer, epoch: int, config: dict[str, Any], metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": safe_config,
            "metrics": metrics,
        },
        path,
    )
    print(f"Saved checkpoint: {path}")


def save_reconstruction_samples(model, dataset, device, output_dir: Path, max_samples: int = 8) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved: list[str] = []
    mean_std = dataset.normalization
    with torch.no_grad():
        for idx in range(min(max_samples, len(dataset))):
            item = dataset[idx]
            motion = item["motion"].unsqueeze(0).to(device)
            recon, _, _, _ = model(motion)
            recon_np = recon.squeeze(0).detach().cpu().numpy().astype(np.float32)
            if dataset.use_normalization and mean_std is not None:
                recon_np = denormalize_sequence(recon_np, *mean_std)
            path = output_dir / f"reconstruction_{idx:03d}_{item['gloss']}.npy"
            np.save(path, recon_np)
            saved.append(str(path))
    return saved


def train_motion_vqvae(
    seq_len: int = 80,
    motion_dim: int = 182,
    batch_size: int = 16,
    epochs: int = 300,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-4,
    latent_dim: int = 256,
    codebook_size: int = 512,
    downsample_factor: int = 4,
    commitment_beta: float = 0.25,
    save_every: int = 25,
    num_workers: int = 0,
    sequence_mode: str = "resample",
    canonicalize_camera: bool = True,
    fix_betas: bool = True,
) -> list[dict[str, float]]:
    device = select_device()
    config = dict(locals())
    config["device"] = str(device)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}

    train_ds = SignAvatarDataset("train", seq_len=seq_len, sequence_mode=sequence_mode, canonicalize_camera=canonicalize_camera, fix_betas=fix_betas)
    val_ds = SignAvatarDataset("val", seq_len=seq_len, sequence_mode=sequence_mode, canonicalize_camera=canonicalize_camera, fix_betas=fix_betas)
    test_ds = SignAvatarDataset("test", seq_len=seq_len, sequence_mode=sequence_mode, canonicalize_camera=canonicalize_camera, fix_betas=fix_betas)
    if len(train_ds) == 0:
        raise RuntimeError("Training dataset is empty. Run prepare_data first.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = MotionVQVAE(seq_len, motion_dim, latent_dim, codebook_size, downsample_factor, commitment_beta).to(device)
    print(f"MotionVQVAE parameters: {count_parameters(model)}")
    loss_fn = WeightedMotionReconstructionLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    for epoch in range(epochs):
        print(f"\nVQ-VAE epoch {epoch + 1}/{epochs}")
        train_metrics = _run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_metrics = _run_epoch(model, val_loader, loss_fn, device) if len(val_ds) else train_metrics
        scheduler.step(val_metrics["loss"])
        row: dict[str, float] = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        print(f"train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f}")

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            _save_checkpoint(CHECKPOINT_V2_DIR / "vqvae_best.pth", model, optimizer, epoch, config, val_metrics)
        if (epoch + 1) % save_every == 0:
            _save_checkpoint(CHECKPOINT_V2_DIR / f"vqvae_epoch_{epoch:03d}.pth", model, optimizer, epoch, config, val_metrics)

    test_metrics = _run_epoch(model, test_loader, loss_fn, device) if len(test_ds) else {"loss": float("inf")}
    _save_checkpoint(CHECKPOINT_V2_DIR / "vqvae_final.pth", model, optimizer, epochs - 1, config, test_metrics)
    save_json(OUTPUT_V2_DIR / "logs" / "vqvae_training_history.json", history)
    save_json(OUTPUT_V2_DIR / "logs" / "vqvae_test_metrics.json", test_metrics)
    saved = save_reconstruction_samples(model, train_ds, device, OUTPUT_V2_DIR / "reconstructions")
    print(f"Saved {len(saved)} reconstruction samples")
    return history


if __name__ == "__main__":
    train_motion_vqvae()
