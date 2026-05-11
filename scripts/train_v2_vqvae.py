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
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=80)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--downsample-factor", type=int, default=4)
    parser.add_argument("--commitment-beta", type=float, default=0.25)
    parser.add_argument("--vq-loss-weight", type=float, default=0.25)
    parser.add_argument("--vq-warmup-epochs", type=int, default=50)
    parser.add_argument("--quantizer-type", choices=["standard", "ema"], default="ema")
    parser.add_argument("--disable-quantization", action="store_true")
    parser.add_argument("--no-dead-code-reset", dest="dead_code_reset", action="store_false", default=True)
    parser.add_argument("--dead-code-threshold", type=int, default=1)
    args = parser.parse_args()
    train_motion_vqvae(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_every=args.save_every,
        num_workers=args.num_workers,
        seq_len=args.seq_len,
        codebook_size=args.codebook_size,
        latent_dim=args.latent_dim,
        downsample_factor=args.downsample_factor,
        commitment_beta=args.commitment_beta,
        quantizer_type=args.quantizer_type,
        vq_loss_weight=args.vq_loss_weight,
        vq_warmup_epochs=args.vq_warmup_epochs,
        disable_quantization=args.disable_quantization,
        dead_code_reset=args.dead_code_reset,
        dead_code_threshold=args.dead_code_threshold,
    )


if __name__ == "__main__":
    main()
