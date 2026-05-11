import json
import random
import argparse
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.config import DATA_DIR, PROCESSED_DIR
from src.data.dataset import CPU_Unpickler, adjust_sequence_length, extract_smplx_sequence


class PrepareDataError(RuntimeError):
    pass


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def load_pickle(path: Path) -> Any:
    try:
        with path.open("rb") as f:
            return CPU_Unpickler(f).load()
    except Exception as exc:
        raise PrepareDataError(f"Failed to load pickle {path}: {exc}") from exc


def _split_paths(paths: list[str], train_ratio: float, val_ratio: float, test_ratio: float, rng: random.Random) -> dict[str, list[str]]:
    if not paths:
        return {"train": [], "val": [], "test": []}

    paths = list(paths)
    rng.shuffle(paths)
    n = len(paths)
    train_end = int(round(n * train_ratio))
    val_end = int(round(n * (train_ratio + val_ratio)))

    train_paths = paths[:train_end]
    val_paths = paths[train_end:val_end]
    test_paths = paths[val_end:]

    if not train_paths:
        if val_paths:
            train_paths.append(val_paths.pop(0))
        elif test_paths:
            train_paths.append(test_paths.pop(0))

    return {"train": train_paths, "val": val_paths, "test": test_paths}


def _build_vocab(glosses: list[str]) -> dict[str, int]:
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for idx, gloss in enumerate(sorted(glosses), start=2):
        vocab[gloss] = idx
    return vocab


def _select_glosses(gloss_to_paths: dict[str, list[str]], max_glosses: int | None, selection_mode: str) -> list[str]:
    if selection_mode == "alphabetical":
        glosses = sorted(gloss_to_paths)
    elif selection_mode == "top_count":
        glosses = sorted(gloss_to_paths, key=lambda gloss: (-len(gloss_to_paths[gloss]), gloss))
    else:
        raise ValueError("selection_mode must be one of: alphabetical, top_count")

    if max_glosses is not None:
        glosses = glosses[:max_glosses]
    return glosses


def _collect_samples(
    index: dict[str, Any],
    max_glosses: int | None,
    max_samples_per_gloss: int | None,
    selection_mode: str,
    rng: random.Random,
) -> dict[str, list[str]]:
    gloss_to_paths = {gloss: list(paths) for gloss, paths in index.get("gloss_to_paths", {}).items()}
    glosses = _select_glosses(gloss_to_paths, max_glosses, selection_mode)

    result: dict[str, list[str]] = {}
    for gloss in glosses:
        paths = list(gloss_to_paths.get(gloss, []))
        rng.shuffle(paths)
        if max_samples_per_gloss is not None:
            paths = paths[:max_samples_per_gloss]
        if paths:
            result[gloss] = paths
    return result


def prepare_data(
    seq_len: int = 60,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    max_glosses: int | None = None,
    max_samples_per_gloss: int | None = None,
    selection_mode: str = "alphabetical",
    sequence_mode: str = "pad_trim",
) -> dict[str, Any]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio, val_ratio and test_ratio must sum to 1.0")
    if selection_mode not in {"alphabetical", "top_count"}:
        raise ValueError("selection_mode must be one of: alphabetical, top_count")
    if sequence_mode not in {"pad_trim", "resample"}:
        raise ValueError("sequence_mode must be one of: pad_trim, resample")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    index_path = PROCESSED_DIR / "index.json"
    if not index_path.exists():
        raise FileNotFoundError("Index file not found. Run build_index first.")

    index = _load_json(index_path)
    gloss_to_paths = _collect_samples(
        index=index,
        max_glosses=max_glosses,
        max_samples_per_gloss=max_samples_per_gloss,
        selection_mode=selection_mode,
        rng=random.Random(seed),
    )
    vocab = _build_vocab(list(gloss_to_paths.keys()))

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": [], "path_to_gloss": {}}
    for gloss, paths in gloss_to_paths.items():
        partition = _split_paths(paths, train_ratio, val_ratio, test_ratio, rng)
        for split_name in ["train", "val", "test"]:
            for path in partition[split_name]:
                splits[split_name].append(path)
                splits["path_to_gloss"][path] = gloss

    train_paths = splits["train"]
    if not train_paths:
        raise PrepareDataError("No training paths generated. Check your split ratios and dataset contents.")

    train_samples = len(train_paths)
    motion_dim = 182
    all_train_frames = []
    for path_str in tqdm(train_paths, desc="Computing normalization stats", unit="file"):
        path = Path(path_str)
        data = load_pickle(path)
        sequence = extract_smplx_sequence(data)
        sequence = adjust_sequence_length(sequence, seq_len, sequence_mode)
        all_train_frames.append(sequence)

    all_train_frames = np.concatenate(all_train_frames, axis=0)
    mean = np.mean(all_train_frames, axis=0).astype(np.float32)
    std = np.std(all_train_frames, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)

    metadata = {
        "vocab_size": len(vocab),
        "num_glosses": len(gloss_to_paths),
        "train_samples": len(splits["train"]),
        "val_samples": len(splits["val"]),
        "test_samples": len(splits["test"]),
        "motion_dimension": motion_dim,
        "seq_len": seq_len,
        "selection_mode": selection_mode,
        "sequence_mode": sequence_mode,
        "selected_glosses": sorted(gloss_to_paths),
        "normalization_stats": str(DATA_DIR / "normalization_stats.json"),
    }

    _save_json(DATA_DIR / "vocab.json", vocab)
    _save_json(DATA_DIR / "splits.json", splits)
    _save_json(DATA_DIR / "normalization_stats.json", {"mean": mean.tolist(), "std": std.tolist()})
    _save_json(PROCESSED_DIR / "prepare_data_summary.json", {**metadata, "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio}})

    print(f"Vocab size: {metadata['vocab_size']}")
    print(f"Number of glosses: {metadata['num_glosses']}")
    print(f"Train samples: {metadata['train_samples']}")
    print(f"Val samples: {metadata['val_samples']}")
    print(f"Test samples: {metadata['test_samples']}")
    print(f"Motion dimension: {metadata['motion_dimension']}")
    print(f"Sequence length: {metadata['seq_len']}")
    print(f"Selection mode: {metadata['selection_mode']}")
    print(f"Sequence mode: {metadata['sequence_mode']}")
    print(f"Selected glosses: {', '.join(metadata['selected_glosses'])}")
    print(f"Saved normalization stats to {DATA_DIR / 'normalization_stats.json'}")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SignAvatar splits, vocab, and normalization stats.")
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-glosses", type=int, default=None)
    parser.add_argument("--max-samples-per-gloss", type=int, default=None)
    parser.add_argument("--selection-mode", choices=["alphabetical", "top_count"], default="alphabetical")
    parser.add_argument("--sequence-mode", choices=["pad_trim", "resample"], default="pad_trim")
    args = parser.parse_args()

    prepare_data(
        seq_len=args.seq_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_glosses=args.max_glosses,
        max_samples_per_gloss=args.max_samples_per_gloss,
        selection_mode=args.selection_mode,
        sequence_mode=args.sequence_mode,
    )
