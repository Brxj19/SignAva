import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import SignAvatarDataset
from src.model_v2.train_token_generator import _build_vqvae_from_checkpoint
from src.model_v2.utils import CHECKPOINT_V2_DIR, OUTPUT_V2_DIR, select_device


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _repeated_token_ratio(tokens: list[int]) -> float:
    counts = Counter(tokens)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return float(repeated / max(len(tokens), 1))


def _transition_count(tokens: list[int]) -> int:
    return sum(1 for left, right in zip(tokens, tokens[1:]) if left != right)


def _summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    gloss_unique: dict[str, list[int]] = defaultdict(list)
    gloss_transition: dict[str, list[int]] = defaultdict(list)
    all_tokens: list[int] = []
    for entry in entries:
        all_tokens.extend(entry["token_ids"])
        gloss_unique[entry["gloss"]].append(entry["unique_token_count"])
        gloss_transition[entry["gloss"]].append(entry["transition_count"])

    token_counts = Counter(all_tokens)
    count = max(len(entries), 1)
    return {
        "num_sequences": len(entries),
        "average_unique_tokens": float(sum(entry["unique_token_count"] for entry in entries) / count),
        "average_repeated_token_ratio": float(sum(entry["repeated_token_ratio"] for entry in entries) / count),
        "average_transition_count": float(sum(entry["transition_count"] for entry in entries) / count),
        "top_10_real_tokens": [{"token": int(token), "count": int(value)} for token, value in token_counts.most_common(10)],
        "per_gloss_average_unique_token_count": {
            gloss: float(sum(values) / len(values)) for gloss, values in sorted(gloss_unique.items())
        },
        "per_gloss_average_transition_count": {
            gloss: float(sum(values) / len(values)) for gloss, values in sorted(gloss_transition.items())
        },
    }


def inspect_real_tokens(split: str = "train", batch_size: int = 16) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")
    device = select_device()
    vqvae_path = CHECKPOINT_V2_DIR / "vqvae_best.pth"
    vqvae = _build_vqvae_from_checkpoint(vqvae_path, device)
    dataset = SignAvatarDataset(
        split,
        seq_len=vqvae.seq_len,
        sequence_mode="resample",
        canonicalize_camera=True,
        fix_betas=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    entries: list[dict[str, Any]] = []
    sample_index = 0
    print(f"Encoding real {split} motions with {vqvae_path}")
    vqvae.eval()
    with torch.no_grad():
        for batch in loader:
            motion = batch["motion"].to(device)
            code_indices, _, _ = vqvae.encode(motion)
            for row_index, tokens_tensor in enumerate(code_indices.detach().cpu()):
                tokens = [int(token) for token in tokens_tensor.tolist()]
                counts = Counter(tokens)
                entries.append(
                    {
                        "sample_index": sample_index,
                        "dataset_split": split,
                        "gloss": str(batch["gloss"][row_index]),
                        "gloss_id": int(batch["gloss_id"][row_index].item()),
                        "original_path": str(batch["path"][row_index]),
                        "token_ids": tokens,
                        "unique_token_count": len(set(tokens)),
                        "most_common_token": int(counts.most_common(1)[0][0]) if counts else None,
                        "repeated_token_ratio": _repeated_token_ratio(tokens),
                        "transition_count": _transition_count(tokens),
                    }
                )
                sample_index += 1

    stats = _summarize(entries)
    sequence_path = OUTPUT_V2_DIR / "evaluation" / f"real_token_sequences_{split}.json"
    stats_path = OUTPUT_V2_DIR / "evaluation" / f"real_token_stats_{split}.json"
    _save_json(sequence_path, entries)
    _save_json(stats_path, stats)

    print("\nV2 real token summary")
    print(f"Split: {split}")
    print(f"Sequences: {stats['num_sequences']}")
    print(f"Average unique tokens: {stats['average_unique_tokens']:.3f}")
    print(f"Average transition count: {stats['average_transition_count']:.3f}")
    print(f"Average repeated token ratio: {stats['average_repeated_token_ratio']:.3f}")
    print(f"Top 10 real tokens: {stats['top_10_real_tokens']}")
    print(f"Saved real token sequences: {sequence_path}")
    print(f"Saved real token stats: {stats_path}")
    return entries, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect real VQ-VAE token sequences from dataset motions.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    inspect_real_tokens(split=args.split, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
