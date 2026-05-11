from pathlib import Path
from typing import Any
from collections import Counter

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
    disable_quantization = config.get("disable_quantization", False)
    if isinstance(disable_quantization, str):
        disable_quantization = disable_quantization.lower() in {"1", "true", "yes", "on"}
    model = MotionVQVAE(
        seq_len=int(config.get("seq_len", 80)),
        motion_dim=int(config.get("motion_dim", 182)),
        latent_dim=int(config.get("latent_dim", 256)),
        codebook_size=int(config.get("codebook_size", 512)),
        downsample_factor=int(config.get("downsample_factor", 4)),
        commitment_beta=float(config.get("commitment_beta", 0.25)),
        quantizer_type=str(config.get("quantizer_type", "standard")),
        disable_quantization=bool(disable_quantization),
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
    if tokens.ndim != 2:
        raise ValueError(f"Expected target tokens shape (batch, token_len), got {tuple(tokens.shape)}")
    start = torch.full((tokens.shape[0], 1), start_token_id, dtype=torch.long, device=tokens.device)
    # Correct next-token teacher forcing:
    # targets  = [t0,  t1, ..., t19]
    # previous = [BOS, t0, ..., t18]
    return torch.cat([start, tokens[:, :-1]], dim=1)


def _transition_diversity_loss(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] < 2:
        return logits.new_tensor(0.0)
    probs = F.softmax(logits, dim=-1)
    similarity = F.cosine_similarity(probs[:, 1:, :], probs[:, :-1, :], dim=-1)
    return similarity.mean()


def _sequence_stats(tokens: torch.Tensor) -> dict[str, float]:
    if tokens.numel() == 0:
        return {"unique_token_count": 0.0, "transition_count": 0.0, "repeated_token_ratio": 0.0}
    unique_counts = []
    transitions = []
    repeated_ratios = []
    for row in tokens.detach().cpu().tolist():
        counts = Counter(row)
        unique_counts.append(len(counts))
        transitions.append(sum(1 for a, b in zip(row, row[1:]) if a != b))
        repeated = sum(count - 1 for count in counts.values() if count > 1)
        repeated_ratios.append(repeated / max(len(row), 1))
    return {
        "unique_token_count": float(sum(unique_counts) / len(unique_counts)),
        "transition_count": float(sum(transitions) / len(transitions)),
        "repeated_token_ratio": float(sum(repeated_ratios) / len(repeated_ratios)),
    }


def _token_metrics(logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0) -> dict[str, float]:
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), label_smoothing=label_smoothing)
    pred = torch.argmax(logits, dim=-1)
    accuracy = (pred == targets).float().mean()
    top5 = torch.topk(logits, k=min(5, logits.shape[-1]), dim=-1).indices
    top5_accuracy = (top5 == targets.unsqueeze(-1)).any(dim=-1).float().mean()
    stats = _sequence_stats(pred)
    return {
        "loss": float(loss.detach().cpu()),
        "cross_entropy": float(loss.detach().cpu()),
        "token_accuracy": float(accuracy.detach().cpu()),
        "top5_token_accuracy": float(top5_accuracy.detach().cpu()),
        "generated_unique_token_count": stats["unique_token_count"],
        "generated_transition_count": stats["transition_count"],
        "generated_repeated_token_ratio": stats["repeated_token_ratio"],
    }


def _filter_dataset_paths(dataset: SignAvatarDataset, max_glosses: int | None, selection_mode: str) -> None:
    if max_glosses is None:
        return
    gloss_to_paths: dict[str, list[Path]] = {}
    for path in dataset.paths:
        gloss = dataset.path_to_gloss.get(str(path))
        if gloss is not None:
            gloss_to_paths.setdefault(gloss, []).append(path)
    if selection_mode == "top_count":
        selected = sorted(gloss_to_paths, key=lambda gloss: (-len(gloss_to_paths[gloss]), gloss))[:max_glosses]
    elif selection_mode in {"alphabetical", "all"}:
        selected = sorted(gloss_to_paths)[:max_glosses]
    else:
        raise ValueError("selection_mode must be one of: alphabetical, top_count, all")
    selected_set = set(selected)
    dataset.paths = [path for path in dataset.paths if dataset.path_to_gloss.get(str(path)) in selected_set]


@torch.no_grad()
def _save_generated_token_diagnostics(
    generator: MotionTokenGenerator,
    vocab: dict[str, int],
    device: torch.device,
    epoch: int,
    max_glosses: int | None = None,
    selection_mode: str = "all",
) -> None:
    glosses = sorted(gloss for gloss in vocab if gloss not in {"<PAD>", "<UNK>"})
    if max_glosses is not None:
        glosses = glosses[:max_glosses] if selection_mode != "top_count" else glosses[:max_glosses]
    entries = []
    for gloss in glosses:
        gloss_id = int(vocab[gloss])
        generated = generator.generate(
            torch.tensor([gloss_id], dtype=torch.long, device=device),
            greedy=True,
            repetition_penalty=1.0,
        ).squeeze(0).detach().cpu()
        tokens = [int(token) for token in generated.tolist()]
        counts = Counter(tokens)
        entries.append(
            {
                "epoch": epoch,
                "gloss": gloss,
                "gloss_id": gloss_id,
                "token_ids": tokens,
                "unique_token_count": len(set(tokens)),
                "transition_count": sum(1 for left, right in zip(tokens, tokens[1:]) if left != right),
                "repeated_token_ratio": sum(count - 1 for count in counts.values() if count > 1) / max(len(tokens), 1),
            }
        )
    if not entries:
        return
    stats = {
        "epoch": epoch,
        "num_glosses": len(entries),
        "average_unique_tokens": sum(entry["unique_token_count"] for entry in entries) / len(entries),
        "average_transition_count": sum(entry["transition_count"] for entry in entries) / len(entries),
        "average_repeated_token_ratio": sum(entry["repeated_token_ratio"] for entry in entries) / len(entries),
    }
    save_json(OUTPUT_V2_DIR / "evaluation" / f"token_generator_generated_sequences_epoch_{epoch:03d}.json", entries)
    save_json(OUTPUT_V2_DIR / "evaluation" / f"token_generator_generated_stats_epoch_{epoch:03d}.json", stats)


def _run_epoch(
    generator,
    vqvae,
    loader,
    device,
    optimizer=None,
    label_smoothing: float = 0.0,
    transition_loss_weight: float = 0.0,
    gloss_aux_weight: float = 0.0,
) -> dict[str, float]:
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
            if previous.shape != targets.shape:
                raise RuntimeError(f"Shifted input shape {tuple(previous.shape)} does not match targets {tuple(targets.shape)}")
            if previous[:, 0].ne(generator.start_token_id).any():
                raise RuntimeError("First previous token must be BOS")

            if is_train:
                optimizer.zero_grad()
            if gloss_aux_weight > 0:
                logits, gloss_logits = generator.forward_with_aux(gloss_id, previous)
            else:
                logits = generator(gloss_id, previous)
                gloss_logits = None
            ce_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), label_smoothing=label_smoothing)
            transition_loss = _transition_diversity_loss(logits)
            loss = ce_loss + transition_loss_weight * transition_loss
            if gloss_logits is not None:
                loss = loss + gloss_aux_weight * F.cross_entropy(gloss_logits, gloss_id)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                optimizer.step()

            metrics = _token_metrics(logits, targets, label_smoothing=label_smoothing)
            metrics["loss"] = float(loss.detach().cpu())
            metrics["transition_loss"] = float(transition_loss.detach().cpu())
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
    save_every: int = 25,
    label_smoothing: float = 0.05,
    transition_loss_weight: float = 0.0,
    gloss_aux_weight: float = 0.0,
    max_glosses: int | None = None,
    selection_mode: str = "all",
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
    _filter_dataset_paths(train_ds, max_glosses=max_glosses, selection_mode=selection_mode)
    _filter_dataset_paths(val_ds, max_glosses=max_glosses, selection_mode=selection_mode)
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
        train_metrics = _run_epoch(
            generator,
            vqvae,
            train_loader,
            device,
            optimizer,
            label_smoothing=label_smoothing,
            transition_loss_weight=transition_loss_weight,
            gloss_aux_weight=gloss_aux_weight,
        )
        val_metrics = _run_epoch(
            generator,
            vqvae,
            val_loader,
            device,
            label_smoothing=label_smoothing,
            transition_loss_weight=transition_loss_weight,
            gloss_aux_weight=gloss_aux_weight,
        ) if len(val_ds) else train_metrics
        scheduler.step(val_metrics["loss"])
        row: dict[str, float] = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        _save_generated_token_diagnostics(generator, vocab, device, epoch, max_glosses=max_glosses, selection_mode=selection_mode)
        print(
            f"train_ce={train_metrics['cross_entropy']:.6f} "
            f"val_acc={val_metrics['token_accuracy']:.4f} "
            f"val_unique={val_metrics['generated_unique_token_count']:.2f} "
            f"val_transitions={val_metrics['generated_transition_count']:.2f}"
        )

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            _save_checkpoint(CHECKPOINT_V2_DIR / "token_generator_best.pth", generator, optimizer, epoch, config, val_metrics)
        if (epoch + 1) % save_every == 0:
            _save_checkpoint(CHECKPOINT_V2_DIR / f"token_generator_epoch_{epoch:03d}.pth", generator, optimizer, epoch, config, val_metrics)

    _save_checkpoint(CHECKPOINT_V2_DIR / "token_generator_final.pth", generator, optimizer, epochs - 1, config, val_metrics)
    save_json(OUTPUT_V2_DIR / "logs" / "token_generator_training_history.json", history)
    return history


if __name__ == "__main__":
    train_token_generator()
