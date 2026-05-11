from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import SignAvatarDataset
from src.model_v2.token_generator import MotionTokenGenerator
from src.model_v2.utils import CHECKPOINT_V2_DIR, OUTPUT_V2_DIR, count_parameters, load_vocab, safe_torch_load, save_json, select_device
from src.model_v2.vqvae import MotionVQVAE


def _build_vqvae_from_checkpoint(checkpoint_path: Path, device: torch.device) -> MotionVQVAE:
    checkpoint = safe_torch_load(checkpoint_path)
    config = checkpoint.get("config", {})
    model = MotionVQVAE(
        seq_len=int(config.get("seq_len", 80)),
        motion_dim=int(config.get("motion_dim", 182)),
        latent_dim=int(config.get("latent_dim", 256)),
        codebook_size=int(config.get("codebook_size", 512)),
        downsample_factor=int(config.get("downsample_factor", 4)),
        commitment_beta=float(config.get("commitment_beta", 0.25)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


@torch.no_grad()
def _encode_batch(vqvae: MotionVQVAE, motion: torch.Tensor) -> torch.Tensor:
    code_indices, _, _ = vqvae.encode(motion)
    return code_indices


def _shift_tokens(tokens: torch.Tensor, start_token_id: int) -> torch.Tensor:
    start = torch.full((tokens.shape[0], 1), start_token_id, dtype=torch.long, device=tokens.device)
    return torch.cat([start, tokens[:, :-1]], dim=1)


def _token_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    pred = torch.argmax(logits, dim=-1)
    accuracy = (pred == targets).float().mean()
    top5 = torch.topk(logits, k=min(5, logits.shape[-1]), dim=-1).indices
    top5_accuracy = (top5 == targets.unsqueeze(-1)).any(dim=-1).float().mean()
    return {
        "loss": float(loss.detach().cpu()),
        "cross_entropy": float(loss.detach().cpu()),
        "token_accuracy": float(accuracy.detach().cpu()),
        "top5_token_accuracy": float(top5_accuracy.detach().cpu()),
    }


def _run_epoch(generator, vqvae, loader, device, optimizer=None) -> dict[str, float]:
    is_train = optimizer is not None
    generator.train(is_train)
    totals: dict[str, float] = {}
    per_gloss_correct: dict[str, float] = {}
    per_gloss_total: dict[str, float] = {}
    batches = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="Training tokens" if is_train else "Evaluating tokens", leave=False):
            motion = batch["motion"].to(device)
            gloss_id = batch["gloss_id"].to(device)
            targets = _encode_batch(vqvae, motion)
            previous = _shift_tokens(targets, generator.start_token_id)

            if is_train:
                optimizer.zero_grad()
            logits = generator(gloss_id, previous)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                optimizer.step()

            metrics = _token_metrics(logits, targets)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            pred = torch.argmax(logits, dim=-1)
            for gloss, correct, total in zip(batch["gloss"], (pred == targets).sum(dim=1), torch.full((targets.shape[0],), targets.shape[1], device=device)):
                per_gloss_correct[gloss] = per_gloss_correct.get(gloss, 0.0) + float(correct.detach().cpu())
                per_gloss_total[gloss] = per_gloss_total.get(gloss, 0.0) + float(total.detach().cpu())
            batches += 1

    if batches == 0:
        return {"loss": float("inf"), "cross_entropy": float("inf"), "token_accuracy": 0.0, "top5_token_accuracy": 0.0}
    out = {key: value / batches for key, value in totals.items()}
    for gloss, total in per_gloss_total.items():
        out[f"per_gloss_token_accuracy/{gloss}"] = per_gloss_correct[gloss] / max(total, 1.0)
    return out


def _save_checkpoint(path: Path, generator, optimizer, epoch: int, config: dict[str, Any], metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    torch.save(
        {
            "model_state_dict": generator.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": safe_config,
            "metrics": metrics,
        },
        path,
    )
    print(f"Saved checkpoint: {path}")


def train_token_generator(
    vqvae_checkpoint: str | Path | None = None,
    epochs: int = 300,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    teacher_forcing: bool = True,
    num_workers: int = 0,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 4,
    dropout: float = 0.1,
    sequence_mode: str = "resample",
    canonicalize_camera: bool = True,
    fix_betas: bool = True,
) -> list[dict[str, float]]:
    if not teacher_forcing:
        raise ValueError("Only teacher_forcing=True is currently implemented.")
    device = select_device()
    vqvae_path = Path(vqvae_checkpoint) if vqvae_checkpoint is not None else CHECKPOINT_V2_DIR / "vqvae_best.pth"
    vqvae = _build_vqvae_from_checkpoint(vqvae_path, device)
    vocab, _ = load_vocab()

    train_ds = SignAvatarDataset("train", seq_len=vqvae.seq_len, sequence_mode=sequence_mode, canonicalize_camera=canonicalize_camera, fix_betas=fix_betas)
    val_ds = SignAvatarDataset("val", seq_len=vqvae.seq_len, sequence_mode=sequence_mode, canonicalize_camera=canonicalize_camera, fix_betas=fix_betas)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    config = dict(locals())
    config["device"] = str(device)
    config["vqvae_checkpoint"] = str(vqvae_path)
    config["seq_len"] = vqvae.seq_len
    config["motion_dim"] = vqvae.motion_dim
    config["token_len"] = vqvae.token_len
    config["codebook_size"] = vqvae.codebook_size
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    config.pop("vqvae", None)
    config.pop("train_ds", None)
    config.pop("val_ds", None)
    config.pop("train_loader", None)
    config.pop("val_loader", None)

    generator = MotionTokenGenerator(
        vocab_size=len(vocab),
        token_len=vqvae.token_len,
        codebook_size=vqvae.codebook_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    print(f"MotionTokenGenerator parameters: {count_parameters(generator)}")
    optimizer = AdamW(generator.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    for epoch in range(epochs):
        print(f"\nToken generator epoch {epoch + 1}/{epochs}")
        train_metrics = _run_epoch(generator, vqvae, train_loader, device, optimizer)
        val_metrics = _run_epoch(generator, vqvae, val_loader, device) if len(val_ds) else train_metrics
        scheduler.step(val_metrics["loss"])
        row: dict[str, float] = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        print(f"train_ce={train_metrics['cross_entropy']:.6f} val_acc={val_metrics['token_accuracy']:.4f}")

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            _save_checkpoint(CHECKPOINT_V2_DIR / "token_generator_best.pth", generator, optimizer, epoch, config, val_metrics)

    _save_checkpoint(CHECKPOINT_V2_DIR / "token_generator_final.pth", generator, optimizer, epochs - 1, config, val_metrics)
    save_json(OUTPUT_V2_DIR / "logs" / "token_generator_training_history.json", history)
    return history


if __name__ == "__main__":
    train_token_generator()
