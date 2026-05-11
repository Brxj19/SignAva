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


def _code_usage_stats(histogram: torch.Tensor) -> dict[str, Any]:
    total = int(histogram.sum().item())
    codebook_size = int(histogram.numel())
    if total == 0:
        return {
            "unique_codes": 0,
            "active_code_pct": 0.0,
            "dead_code_count": codebook_size,
            "code_usage_entropy": 0.0,
            "top_10_codes": [],
            "code_usage_histogram": [0 for _ in range(codebook_size)],
        }
    probs = histogram.to(torch.float64) / float(total)
    active = histogram > 0
    entropy = -torch.sum(probs[active] * torch.log(probs[active])).item()
    top_values, top_indices = torch.topk(histogram, k=min(10, codebook_size))
    return {
        "unique_codes": int(active.sum().item()),
        "active_code_pct": float(active.sum().item() / codebook_size * 100.0),
        "dead_code_count": int((~active).sum().item()),
        "code_usage_entropy": float(entropy),
        "top_10_codes": [
            {"code": int(index.item()), "count": int(value.item())}
            for value, index in zip(top_values, top_indices)
            if int(value.item()) > 0
        ],
        "code_usage_histogram": [int(value) for value in histogram.tolist()],
    }


def _run_epoch(
    model,
    loader,
    loss_fn,
    device,
    optimizer=None,
    dead_code_reset: bool = False,
    dead_code_threshold: int = 1,
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    rows: list[dict[str, float]] = []
    histogram = torch.zeros(model.codebook_size, dtype=torch.long)
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="Training" if is_train else "Evaluating", leave=False):
            motion = batch["motion"].to(device)
            if is_train:
                optimizer.zero_grad()
            recon, code_indices, vq_loss, perplexity = model(motion)
            metrics = loss_fn(recon, motion, vq_loss=vq_loss, perplexity=perplexity)
            if is_train:
                metrics["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            if not model.disable_quantization:
                histogram += torch.bincount(code_indices.detach().cpu().reshape(-1), minlength=model.codebook_size)
            rows.append({key: float(value.detach().cpu()) for key, value in metrics.items()})
    output: dict[str, Any] = _mean_metrics(rows)
    output.update(_code_usage_stats(histogram))
    output["codes_reset"] = 0
    if is_train and dead_code_reset and not model.disable_quantization:
        dead_codes = torch.nonzero(histogram <= dead_code_threshold, as_tuple=False).reshape(-1).to(device)
        output["codes_reset"] = model.reset_dead_codes(dead_codes)
    return output


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
    codebook_size: int = 256,
    downsample_factor: int = 4,
    commitment_beta: float = 0.25,
    quantizer_type: str = "ema",
    vq_loss_weight: float = 0.25,
    vq_warmup_epochs: int = 50,
    disable_quantization: bool = False,
    dead_code_reset: bool = True,
    dead_code_threshold: int = 1,
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

    model = MotionVQVAE(
        seq_len=seq_len,
        motion_dim=motion_dim,
        latent_dim=latent_dim,
        codebook_size=codebook_size,
        downsample_factor=downsample_factor,
        commitment_beta=commitment_beta,
        quantizer_type=quantizer_type,
        disable_quantization=disable_quantization,
    ).to(device)
    print(f"MotionVQVAE parameters: {count_parameters(model)}")
    loss_fn = WeightedMotionReconstructionLoss(vq_loss_weight=vq_loss_weight).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    for epoch in range(epochs):
        print(f"\nVQ-VAE epoch {epoch + 1}/{epochs}")
        if disable_quantization:
            current_vq_weight = 0.0
        elif vq_warmup_epochs > 0:
            start_weight = min(0.05, vq_loss_weight)
            progress = min(1.0, float(epoch + 1) / float(vq_warmup_epochs))
            current_vq_weight = start_weight + (vq_loss_weight - start_weight) * progress
        else:
            current_vq_weight = vq_loss_weight
        loss_fn.vq_loss_weight = current_vq_weight
        train_metrics = _run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            optimizer,
            dead_code_reset=dead_code_reset,
            dead_code_threshold=dead_code_threshold,
        )
        val_metrics = _run_epoch(model, val_loader, loss_fn, device) if len(val_ds) else train_metrics
        scheduler.step(val_metrics["loss"])
        row: dict[str, Any] = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "vq_loss_weight": current_vq_weight}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        print(
            f"train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f} "
            f"train_recon={train_metrics['recon_loss']:.6f} val_recon={val_metrics['recon_loss']:.6f} "
            f"perplexity={val_metrics['perplexity']:.3f} active={val_metrics['active_code_pct']:.2f}% "
            f"vq_weight={current_vq_weight:.4f} reset={train_metrics['codes_reset']}"
        )

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
