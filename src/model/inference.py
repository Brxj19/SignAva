import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import CHECKPOINT_DIR, DATA_DIR, OUTPUT_DIR
from src.model.architecture import SignMotionGenerator


DEFAULT_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pth"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "generated"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "generated_smplx.npy"
DEFAULT_METADATA_PATH = DEFAULT_OUTPUT_DIR / "generated_metadata.json"


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _safe_torch_load(checkpoint_path: Path) -> dict[str, Any]:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def load_checkpoint(checkpoint_path: str | Path | None = None) -> tuple[SignMotionGenerator, dict[str, Any], torch.device]:
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else DEFAULT_CHECKPOINT_PATH

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint missing: {checkpoint_path}. Expected a trained model at "
            "checkpoints/best_model.pth from Milestone 4."
        )

    checkpoint = _safe_torch_load(checkpoint_path)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint format at {checkpoint_path}: expected a dictionary.")

    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise KeyError(
            f"Model config missing in checkpoint: {checkpoint_path}. "
            "Retrain or save a checkpoint that includes the 'config' field."
        )

    vocab_size = checkpoint.get("vocab_size")
    if vocab_size is None:
        raise KeyError(
            f"vocab_size missing in checkpoint: {checkpoint_path}. "
            "Milestone 5 needs the saved vocabulary size to rebuild the model."
        )

    model_state_dict = checkpoint.get("model_state_dict")
    if model_state_dict is None:
        raise KeyError(f"model_state_dict missing in checkpoint: {checkpoint_path}.")

    model = SignMotionGenerator(
        vocab_size=int(vocab_size),
        seq_len=int(config.get("seq_len", 60)),
        motion_dim=int(config.get("motion_dim", 182)),
        d_model=int(config.get("d_model", 256)),
        nhead=int(config.get("nhead", 8)),
        num_layers=int(config.get("num_layers", 4)),
        dropout=float(config.get("dropout", 0.1)),
    )
    model.load_state_dict(model_state_dict)
    model.eval()

    device = _select_device()
    model = model.to(device)

    checkpoint["_resolved_checkpoint_path"] = str(checkpoint_path)
    return model, checkpoint, device


def load_vocab() -> tuple[dict[str, int], dict[int, str]]:
    vocab_path = DATA_DIR / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Vocab missing: {vocab_path}. Run Milestone 2 data preparation before inference."
        )

    with vocab_path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)

    if not isinstance(vocab, dict) or not vocab:
        raise ValueError(f"Invalid vocab at {vocab_path}: expected a non-empty JSON object.")

    try:
        vocab = {str(gloss): int(gloss_id) for gloss, gloss_id in vocab.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid vocab IDs in {vocab_path}: expected integer IDs.") from exc

    id_to_gloss = {gloss_id: gloss for gloss, gloss_id in vocab.items()}
    return vocab, id_to_gloss


def load_normalization_stats() -> tuple[np.ndarray, np.ndarray]:
    stats_path = DATA_DIR / "normalization_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Normalization stats missing: {stats_path}. Run Milestone 2 data preparation first."
        )

    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    if "mean" not in stats or "std" not in stats:
        raise KeyError(
            f"Normalization stats missing required keys in {stats_path}. Expected 'mean' and 'std'."
        )

    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    if mean.shape != std.shape:
        raise ValueError(f"Normalization mean/std shape mismatch: mean={mean.shape}, std={std.shape}.")

    return mean, std


def text_to_gloss_id(text: str, vocab: dict[str, int]) -> tuple[str, int]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text is empty. Provide a gloss such as 'about'.")

    gloss = text.lower().strip()
    if gloss in vocab:
        return gloss, int(vocab[gloss])

    if "<UNK>" not in vocab:
        raise KeyError("Unknown gloss and <UNK> token missing from vocab.json.")

    warnings.warn(
        f"Unknown gloss '{gloss}'. Falling back to <UNK>. "
        f"Use list_available_glosses() to see supported glosses.",
        stacklevel=2,
    )
    return "<UNK>", int(vocab["<UNK>"])


def _resolve_output_paths(output_path: str | Path | None) -> tuple[Path, Path]:
    if output_path is None:
        motion_path = DEFAULT_OUTPUT_PATH
    else:
        motion_path = Path(output_path)
        if motion_path.suffix.lower() != ".npy":
            motion_path = motion_path / "generated_smplx.npy"

    metadata_path = motion_path.with_name("generated_metadata.json")
    return motion_path, metadata_path


def generate_motion(
    text: str,
    checkpoint_path: str | Path | None = None,
    output_path: str | Path | None = None,
    denormalize: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    model, checkpoint, device = load_checkpoint(checkpoint_path)
    vocab, _ = load_vocab()
    matched_gloss, gloss_id = text_to_gloss_id(text, vocab)

    gloss_tensor = torch.tensor([gloss_id], dtype=torch.long, device=device)
    with torch.no_grad():
        generated = model(gloss_tensor)

    expected_shape = (1, 60, 182)
    if tuple(generated.shape) != expected_shape:
        raise RuntimeError(f"Unexpected model output shape: {tuple(generated.shape)}. Expected {expected_shape}.")

    motion = generated.squeeze(0).detach().cpu().numpy().astype(np.float32)

    if denormalize:
        mean, std = load_normalization_stats()
        if mean.shape != (motion.shape[-1],) or std.shape != (motion.shape[-1],):
            raise ValueError(
                "Normalization stats shape mismatch: "
                f"motion_dim={motion.shape[-1]}, mean={mean.shape}, std={std.shape}."
            )
        motion = motion * std + mean

    motion_path, metadata_path = _resolve_output_paths(output_path)
    motion_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(motion_path, motion)

    metadata = {
        "input_text": text,
        "matched_gloss": matched_gloss,
        "gloss_id": gloss_id,
        "checkpoint_path": checkpoint.get("_resolved_checkpoint_path", str(checkpoint_path or DEFAULT_CHECKPOINT_PATH)),
        "output_shape": list(motion.shape),
        "denormalized": denormalize,
        "device": str(device),
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return motion, metadata


def list_available_glosses() -> list[str]:
    vocab, _ = load_vocab()
    glosses = sorted(gloss for gloss in vocab if gloss not in {"<PAD>", "<UNK>"})
    print("Available glosses:")
    for gloss in glosses:
        print(f"- {gloss}")
    return glosses


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SMPL-X motion from gloss text.")
    parser.add_argument("--text", type=str, help="Gloss text to generate, for example: about")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a trained checkpoint.")
    parser.add_argument("--output-path", type=str, default=None, help="Output .npy path or output directory.")
    parser.add_argument("--no-denormalize", action="store_true", help="Save normalized model output.")
    parser.add_argument("--list-glosses", action="store_true", help="Print supported glosses and exit.")
    args = parser.parse_args()

    if args.list_glosses:
        list_available_glosses()
        return

    if not args.text:
        parser.error("--text is required unless --list-glosses is used.")

    motion, metadata = generate_motion(
        args.text,
        checkpoint_path=args.checkpoint,
        output_path=args.output_path,
        denormalize=not args.no_denormalize,
    )
    print(f"Generated motion shape: {motion.shape}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
