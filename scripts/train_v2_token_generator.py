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
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--transition-loss-weight", type=float, default=0.0)
    parser.add_argument("--gloss-aux-weight", type=float, default=0.0)
    parser.add_argument("--max-glosses", type=int, default=None)
    parser.add_argument("--selection-mode", choices=["all", "alphabetical", "top_count"], default="all")
    args = parser.parse_args()
    train_token_generator(
        vqvae_checkpoint=args.vqvae_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        dropout=args.dropout,
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead,
        save_every=args.save_every,
        label_smoothing=args.label_smoothing,
        transition_loss_weight=args.transition_loss_weight,
        gloss_aux_weight=args.gloss_aux_weight,
        max_glosses=args.max_glosses,
        selection_mode=args.selection_mode,
    )


if __name__ == "__main__":
    main()
