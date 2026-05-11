import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.dataset import denormalize_sequence
from src.config import DATA_DIR
from src.model_v2.token_generator import MotionTokenGenerator
from src.model_v2.train_token_generator import _build_vqvae_from_checkpoint
from src.model_v2.utils import CHECKPOINT_V2_DIR, OUTPUT_V2_DIR, load_vocab, safe_filename, safe_torch_load, select_device


def _load_normalization() -> tuple[np.ndarray, np.ndarray]:
    with (DATA_DIR / "normalization_stats.json").open("r", encoding="utf-8") as handle:
        stats = json.load(handle)
    return np.asarray(stats["mean"], dtype=np.float32), np.asarray(stats["std"], dtype=np.float32)


def _load_generator(path: Path, vocab_size: int, device: torch.device) -> MotionTokenGenerator:
    checkpoint = safe_torch_load(path)
    config = checkpoint.get("config", {})
    generator = MotionTokenGenerator(
        vocab_size=vocab_size,
        token_len=int(config.get("token_len", 20)),
        codebook_size=int(config.get("codebook_size", 512)),
        d_model=int(config.get("d_model", 256)),
        nhead=int(config.get("nhead", 8)),
        num_layers=int(config.get("num_layers", 4)),
        dropout=float(config.get("dropout", 0.1)),
    )
    missing, unexpected = generator.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing:
        print(f"Warning: token generator checkpoint missing keys initialized from defaults: {missing}")
    if unexpected:
        print(f"Warning: token generator checkpoint has unexpected keys ignored: {unexpected}")
    return generator.to(device).eval()


def generate_motion_v2(
    text_or_gloss: str,
    checkpoint_dir: str | Path = "checkpoints_v2",
    output_path: str | Path | None = None,
    denormalize: bool = True,
    greedy: bool = True,
    temperature: float = 1.0,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 0,
    top_k: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    device = select_device()
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = CHECKPOINT_V2_DIR.parent / checkpoint_dir
    vqvae_path = checkpoint_dir / "vqvae_best.pth"
    generator_path = checkpoint_dir / "token_generator_best.pth"

    vocab, _ = load_vocab()
    gloss = text_or_gloss.lower().strip()
    gloss_id = vocab.get(gloss, vocab.get("<UNK>"))
    if gloss_id is None:
        raise KeyError(f"Gloss '{text_or_gloss}' not found and <UNK> is missing from vocab.json")
    if gloss not in vocab:
        gloss = "<UNK>"

    vqvae = _build_vqvae_from_checkpoint(vqvae_path, device)
    generator = _load_generator(generator_path, len(vocab), device)

    gloss_tensor = torch.tensor([gloss_id], dtype=torch.long, device=device)
    with torch.no_grad():
        code_indices = generator.generate(
            gloss_tensor,
            temperature=temperature,
            greedy=greedy,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            top_k=top_k,
        )
        motion_tensor = vqvae.decode(code_indices=code_indices)
    motion = motion_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
    if denormalize:
        motion = denormalize_sequence(motion, *_load_normalization())

    if motion.shape != (vqvae.seq_len, vqvae.motion_dim):
        raise RuntimeError(f"Unexpected generated motion shape {motion.shape}; expected {(vqvae.seq_len, vqvae.motion_dim)}")

    safe = safe_filename(gloss)
    motion_path = Path(output_path) if output_path is not None else OUTPUT_V2_DIR / "generated" / f"{safe}_v2_smplx.npy"
    if motion_path.suffix.lower() != ".npy":
        motion_path = motion_path / f"{safe}_v2_smplx.npy"
    metadata_path = motion_path.with_name(f"{safe}_v2_metadata.json")
    motion_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(motion_path, motion)

    metadata = {
        "gloss": gloss,
        "gloss_id": int(gloss_id),
        "seq_len": int(vqvae.seq_len),
        "motion_dim": int(vqvae.motion_dim),
        "token_len": int(vqvae.token_len),
        "codebook_size": int(vqvae.codebook_size),
        "checkpoints": {
            "vqvae": str(vqvae_path),
            "token_generator": str(generator_path),
        },
        "output_path": str(motion_path),
        "denormalized": denormalize,
        "decode": {
            "strategy": "greedy" if greedy else "sample",
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "top_k": top_k,
        },
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return motion, metadata


if __name__ == "__main__":
    generate_motion_v2("about")
