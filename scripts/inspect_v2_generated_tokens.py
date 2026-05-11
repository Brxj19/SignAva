import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.data.dataset import denormalize_sequence
from src.model_v2.inference_v2 import _load_generator, _load_normalization
from src.model_v2.train_token_generator import _build_vqvae_from_checkpoint
from src.model_v2.utils import CHECKPOINT_V2_DIR, OUTPUT_V2_DIR, load_vocab, safe_filename, select_device


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _repeated_token_ratio(tokens: list[int]) -> float:
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return float(repeated / len(tokens))


def _transition_count(tokens: list[int]) -> int:
    return sum(1 for left, right in zip(tokens, tokens[1:]) if left != right)


def _summarize_sequences(entries: list[dict[str, Any]]) -> dict[str, Any]:
    if not entries:
        return {
            "num_sequences": 0,
            "average_unique_tokens": 0.0,
            "average_transition_count": 0.0,
            "average_repeated_token_ratio": 0.0,
            "top_10_tokens": [],
        }
    all_tokens = [token for entry in entries for token in entry["token_ids"]]
    counts = Counter(all_tokens)
    return {
        "num_sequences": len(entries),
        "average_unique_tokens": float(sum(entry["unique_token_count"] for entry in entries) / len(entries)),
        "average_transition_count": float(sum(entry["transition_count"] for entry in entries) / len(entries)),
        "average_repeated_token_ratio": float(sum(entry["repeated_token_ratio"] for entry in entries) / len(entries)),
        "top_10_tokens": [{"token": int(token), "count": int(count)} for token, count in counts.most_common(10)],
    }


def _hamming_similarity(left: list[int], right: list[int]) -> float:
    if len(left) != len(right):
        raise ValueError("Token sequences must have the same length for Hamming similarity")
    if not left:
        return 1.0
    matches = sum(1 for a, b in zip(left, right) if a == b)
    return float(matches / len(left))


def inspect_v2_generated_tokens(save_motion: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    device = select_device()
    vqvae_path = CHECKPOINT_V2_DIR / "vqvae_best.pth"
    generator_path = CHECKPOINT_V2_DIR / "token_generator_best.pth"
    if not vqvae_path.exists():
        raise FileNotFoundError(f"Missing VQ-VAE checkpoint: {vqvae_path}")
    if not generator_path.exists():
        raise FileNotFoundError(f"Missing token generator checkpoint: {generator_path}")

    vocab, _ = load_vocab()
    glosses = sorted(gloss for gloss in vocab if gloss not in {"<PAD>", "<UNK>"})
    vqvae = _build_vqvae_from_checkpoint(vqvae_path, device)
    generator = _load_generator(generator_path, len(vocab), device)
    mean, std = _load_normalization()

    generated_dir = OUTPUT_V2_DIR / "generated" / "token_diagnostic"
    sequence_entries: list[dict[str, Any]] = []
    all_tokens: list[int] = []

    print(f"Inspecting generated tokens for {len(glosses)} glosses")
    print(f"VQ-VAE checkpoint: {vqvae_path}")
    print(f"Token generator checkpoint: {generator_path}")

    for gloss in glosses:
        gloss_id = int(vocab[gloss])
        gloss_tensor = torch.tensor([gloss_id], dtype=torch.long, device=device)
        with torch.no_grad():
            code_indices = generator.generate(gloss_tensor, greedy=True, repetition_penalty=1.0)
            motion_tensor = vqvae.decode(code_indices=code_indices)

        token_ids = [int(token) for token in code_indices.squeeze(0).detach().cpu().tolist()]
        all_tokens.extend(token_ids)
        counts = Counter(token_ids)
        most_common_token = counts.most_common(1)[0][0] if counts else None
        motion_path = None
        if save_motion:
            motion = motion_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
            motion = denormalize_sequence(motion, mean, std)
            safe = safe_filename(gloss)
            motion_path = generated_dir / f"{safe}_tokens_diag_smplx.npy"
            motion_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(motion_path, motion)

        sequence_entries.append(
            {
                "gloss": gloss,
                "gloss_id": gloss_id,
                "token_ids": token_ids,
                "unique_token_count": len(set(token_ids)),
                "most_common_token": most_common_token,
                "repeated_token_ratio": _repeated_token_ratio(token_ids),
                "transition_count": _transition_count(token_ids),
                "generated_motion_path": str(motion_path) if motion_path is not None else None,
            }
        )

    sequence_path = OUTPUT_V2_DIR / "evaluation" / "generated_token_sequences.json"
    _save_json(sequence_path, sequence_entries)
    stats_payload = _summarize_sequences(sequence_entries)
    stats_path = OUTPUT_V2_DIR / "evaluation" / "generated_token_stats.json"
    _save_json(stats_path, stats_payload)

    pair_rows: list[dict[str, Any]] = []
    identical_pairs: list[dict[str, Any]] = []
    for left, right in combinations(sequence_entries, 2):
        left_tokens = left["token_ids"]
        right_tokens = right["token_ids"]
        exact_match_count = sum(1 for a, b in zip(left_tokens, right_tokens) if a == b)
        hamming = _hamming_similarity(left_tokens, right_tokens)
        row = {
            "gloss_a": left["gloss"],
            "gloss_b": right["gloss"],
            "gloss_id_a": left["gloss_id"],
            "gloss_id_b": right["gloss_id"],
            "exact_match_count": exact_match_count,
            "token_length": len(left_tokens),
            "hamming_similarity": hamming,
        }
        pair_rows.append(row)
        if left_tokens == right_tokens:
            identical_pairs.append(row)

    pair_rows.sort(key=lambda row: (-row["hamming_similarity"], row["gloss_a"], row["gloss_b"]))
    average_similarity = float(sum(row["hamming_similarity"] for row in pair_rows) / len(pair_rows)) if pair_rows else 0.0
    token_counts = Counter(all_tokens)
    similarity_payload = {
        "num_glosses": len(sequence_entries),
        "token_length": int(vqvae.token_len),
        "average_hamming_similarity": average_similarity,
        "most_similar_gloss_pairs": pair_rows[:20],
        "identical_token_sequences": identical_pairs,
        "identical_token_sequence_count": len(identical_pairs),
        "top_10_generated_tokens": [
            {"token": int(token), "count": int(count)}
            for token, count in token_counts.most_common(10)
        ],
    }
    similarity_path = OUTPUT_V2_DIR / "evaluation" / "generated_token_similarity.json"
    _save_json(similarity_path, similarity_payload)

    avg_unique = float(
        sum(entry["unique_token_count"] for entry in sequence_entries) / len(sequence_entries)
    ) if sequence_entries else 0.0
    print("\nV2 generated token diversity summary")
    print(f"Glosses: {len(sequence_entries)}")
    print(f"Average unique tokens per sequence: {avg_unique:.3f}")
    print(f"Average transition count: {stats_payload['average_transition_count']:.3f}")
    print(f"Average repeated token ratio: {stats_payload['average_repeated_token_ratio']:.3f}")
    print(f"Average pairwise token similarity: {average_similarity:.3f}")
    print(f"Identical token sequence pairs: {len(identical_pairs)}")
    print(f"Top 10 generated tokens: {similarity_payload['top_10_generated_tokens']}")
    print(f"Saved token sequences: {sequence_path}")
    print(f"Saved token stats: {stats_path}")
    print(f"Saved token similarity: {similarity_path}")
    return sequence_entries, similarity_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect V2 generated token diversity across glosses.")
    parser.add_argument("--no-save-motion", action="store_true", help="Skip saving decoded motion .npy files.")
    args = parser.parse_args()
    inspect_v2_generated_tokens(save_motion=not args.no_save_motion)


if __name__ == "__main__":
    main()
