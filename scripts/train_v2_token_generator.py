import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_v2.train_token_generator import train_token_generator


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the V2 gloss-to-motion-token generator.")
    parser.add_argument("--vqvae-checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    train_token_generator(
        vqvae_checkpoint=args.vqvae_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
