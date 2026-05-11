import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import SignAvatarDataset
from src.model_v2.losses import WeightedMotionReconstructionLoss
from src.model_v2.train_token_generator import _build_vqvae_from_checkpoint, _run_epoch as run_token_epoch
from src.model_v2.train_token_generator import train_token_generator
from src.model_v2.train_vqvae import _run_epoch as run_vqvae_epoch
from src.model_v2.token_generator import MotionTokenGenerator
from src.model_v2.utils import CHECKPOINT_V2_DIR, OUTPUT_V2_DIR, load_vocab, safe_torch_load, save_json, select_device


def evaluate_vqvae(split: str = "test", batch_size: int = 16) -> dict:
    device = select_device()
    vqvae = _build_vqvae_from_checkpoint(CHECKPOINT_V2_DIR / "vqvae_best.pth", device)
    dataset = SignAvatarDataset(split, seq_len=vqvae.seq_len, sequence_mode="resample", canonicalize_camera=True, fix_betas=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    metrics = run_vqvae_epoch(vqvae, loader, WeightedMotionReconstructionLoss().to(device), device)
    save_json(OUTPUT_V2_DIR / "evaluation" / f"vqvae_{split}_metrics.json", metrics)
    return metrics


def evaluate_token_generator(split: str = "test", batch_size: int = 16) -> dict:
    device = select_device()
    vqvae = _build_vqvae_from_checkpoint(CHECKPOINT_V2_DIR / "vqvae_best.pth", device)
    vocab, _ = load_vocab()
    checkpoint = safe_torch_load(CHECKPOINT_V2_DIR / "token_generator_best.pth")
    config = checkpoint.get("config", {})
    generator = MotionTokenGenerator(
        vocab_size=len(vocab),
        token_len=int(config.get("token_len", vqvae.token_len)),
        codebook_size=int(config.get("codebook_size", vqvae.codebook_size)),
        d_model=int(config.get("d_model", 256)),
        nhead=int(config.get("nhead", 8)),
        num_layers=int(config.get("num_layers", 4)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    generator.load_state_dict(checkpoint["model_state_dict"])
    dataset = SignAvatarDataset(split, seq_len=vqvae.seq_len, sequence_mode="resample", canonicalize_camera=True, fix_betas=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    metrics = run_token_epoch(generator, vqvae, loader, device)
    save_json(OUTPUT_V2_DIR / "evaluation" / f"token_generator_{split}_metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V2 VQ-VAE and token generator.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-token", action="store_true")
    args = parser.parse_args()
    results = {"vqvae": evaluate_vqvae(args.split, args.batch_size)}
    if not args.skip_token:
        results["token_generator"] = evaluate_token_generator(args.split, args.batch_size)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
