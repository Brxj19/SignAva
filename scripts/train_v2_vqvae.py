import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_v2.train_vqvae import train_motion_vqvae


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the V2 SignVAE-inspired motion VQ-VAE.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=80)
    args = parser.parse_args()
    train_motion_vqvae(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_every=args.save_every,
        num_workers=args.num_workers,
        seq_len=args.seq_len,
    )


if __name__ == "__main__":
    main()
