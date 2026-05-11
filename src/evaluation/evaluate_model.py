import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CHECKPOINT_DIR, OUTPUT_DIR
from src.data.dataset import SignAvatarDataset
from src.model.architecture import SignMotionGenerator
from src.model.loss import WeightedSignMotionLoss
from src.model.train import select_device


def load_model_from_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[SignMotionGenerator, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config")
    vocab_size = checkpoint.get("vocab_size")
    if not isinstance(config, dict):
        raise KeyError(f"Checkpoint is missing model config: {checkpoint_path}")
    if vocab_size is None:
        raise KeyError(f"Checkpoint is missing vocab_size: {checkpoint_path}")

    model = SignMotionGenerator(
        vocab_size=int(vocab_size),
        seq_len=int(config.get("seq_len", 60)),
        motion_dim=int(config.get("motion_dim", 182)),
        d_model=int(config.get("d_model", 256)),
        nhead=int(config.get("nhead", 8)),
        num_layers=int(config.get("num_layers", 4)),
        dropout=float(config.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def evaluate_model(
    split: str = "test",
    checkpoint_path: str | Path = CHECKPOINT_DIR / "best_model.pth",
    batch_size: int = 8,
    num_workers: int = 0,
    sequence_mode: str | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")

    device = select_device()
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)
    config = checkpoint.get("config", {})
    seq_len = int(config.get("seq_len", 60))
    sequence_mode = sequence_mode or config.get("sequence_mode", "pad_trim")

    dataset = SignAvatarDataset(split=split, seq_len=seq_len, sequence_mode=sequence_mode)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    loss_fn = WeightedSignMotionLoss().to(device)

    totals: dict[str, float] = {}
    num_batches = 0
    num_samples = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {split}", leave=False):
            gloss_id = batch["gloss_id"].to(device)
            motion = batch["motion"].to(device)
            pred_motion = model(gloss_id)
            loss_dict = loss_fn(pred_motion, motion)

            batch_size_actual = gloss_id.shape[0]
            for key, value in loss_dict.items():
                totals[key] = totals.get(key, 0.0) + value.item() * batch_size_actual
            num_batches += 1
            num_samples += batch_size_actual

    if num_samples == 0:
        raise RuntimeError(f"No samples found for split: {split}")

    metrics = {key: value / num_samples for key, value in totals.items()}
    report = {
        "split": split,
        "checkpoint_path": str(checkpoint_path),
        "num_samples": num_samples,
        "num_batches": num_batches,
        "sequence_mode": sequence_mode,
        "seq_len": seq_len,
        "metrics": metrics,
    }

    output_path = Path(output_path) if output_path is not None else OUTPUT_DIR / "evaluation" / "evaluation_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved evaluation metrics to {output_path}")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained SignMotionGenerator checkpoint.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "best_model.pth"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sequence-mode", choices=["pad_trim", "resample"], default=None)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    evaluate_model(
        split=args.split,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sequence_mode=args.sequence_mode,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
