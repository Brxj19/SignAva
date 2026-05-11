import io
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import DATA_DIR


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(
                io.BytesIO(b),
                map_location=torch.device("cpu"),
                weights_only=False,
            )
        return super().find_class(module, name)


def load_pickle(path: Path) -> Any:
    try:
        with path.open("rb") as f:
            return CPU_Unpickler(f).load()
    except Exception as exc:
        raise RuntimeError(f"Failed to load pickle file {path}: {exc}") from exc


def extract_smplx_sequence(data: Any) -> np.ndarray:
    candidate = None
    if isinstance(data, dict):
        candidate = data.get("smplx")
    elif hasattr(data, "smplx"):
        candidate = getattr(data, "smplx")

    if candidate is None:
        raise ValueError("No 'smplx' key found in pickle data")

    if isinstance(candidate, torch.Tensor):
        candidate = candidate.detach().cpu().numpy()
    if isinstance(candidate, np.ndarray):
        sequence = candidate
    elif isinstance(candidate, list):
        sequence = np.asarray(candidate, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported smplx data type: {type(candidate).__name__}")

    if sequence.ndim != 2:
        raise ValueError(f"Expected smplx sequence to be 2D, got shape {sequence.shape}")
    if sequence.shape[1] != 182:
        raise ValueError(f"Expected smplx sequence width 182, got {sequence.shape[1]}")

    return sequence.astype(np.float32)


def pad_or_trim_sequence(sequence: np.ndarray, seq_len: int) -> np.ndarray:
    length, width = sequence.shape
    if width != 182:
        raise ValueError(f"Expected sequence width 182, got {width}")
    if length == seq_len:
        return sequence
    if length > seq_len:
        return sequence[:seq_len, :]

    padded = np.zeros((seq_len, width), dtype=np.float32)
    padded[:length] = sequence
    return padded


def resample_sequence(sequence: np.ndarray, seq_len: int) -> np.ndarray:
    length, width = sequence.shape
    if width != 182:
        raise ValueError(f"Expected sequence width 182, got {width}")
    if length == seq_len:
        return sequence.astype(np.float32)
    if length <= 0:
        raise ValueError("Cannot resample an empty sequence")
    if length == 1:
        return np.repeat(sequence.astype(np.float32), seq_len, axis=0)

    source_positions = np.linspace(0.0, 1.0, num=length, dtype=np.float32)
    target_positions = np.linspace(0.0, 1.0, num=seq_len, dtype=np.float32)
    resampled = np.empty((seq_len, width), dtype=np.float32)
    for dim in range(width):
        resampled[:, dim] = np.interp(target_positions, source_positions, sequence[:, dim]).astype(np.float32)
    return resampled


def adjust_sequence_length(sequence: np.ndarray, seq_len: int, sequence_mode: str = "pad_trim") -> np.ndarray:
    if sequence_mode == "pad_trim":
        return pad_or_trim_sequence(sequence, seq_len)
    if sequence_mode == "resample":
        return resample_sequence(sequence, seq_len)
    raise ValueError("sequence_mode must be one of: pad_trim, resample")


def normalize_sequence(sequence: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if sequence.shape[1] != mean.shape[0] or mean.shape != std.shape:
        raise ValueError("Mean/std shapes are incompatible with motion sequence")
    return (sequence - mean) / std


def denormalize_sequence(sequence: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if sequence.shape[1] != mean.shape[0] or mean.shape != std.shape:
        raise ValueError("Mean/std shapes are incompatible with motion sequence")
    return sequence * std + mean


class SignAvatarDataset(Dataset):
    def __init__(
        self,
        split: str,
        seq_len: int = 60,
        project_root: Path | None = None,
        use_normalization: bool = True,
        max_glosses: int | None = None,
        max_samples_per_gloss: int | None = None,
        sequence_mode: str = "pad_trim",
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")
        if sequence_mode not in {"pad_trim", "resample"}:
            raise ValueError("sequence_mode must be one of: pad_trim, resample")

        self.seq_len = seq_len
        self.sequence_mode = sequence_mode
        self.split = split
        self.use_normalization = use_normalization
        self.project_root = Path(project_root) if project_root is not None else DATA_DIR.parent
        self.data_dir = self.project_root / "data"

        self.vocab = self._load_json(self.data_dir / "vocab.json")
        self.splits = self._load_json(self.data_dir / "splits.json")

        if self.split not in self.splits or "path_to_gloss" not in self.splits:
            raise RuntimeError("Split data is missing or malformed in splits.json")

        self.path_to_gloss = self.splits["path_to_gloss"]
        self.paths = [Path(p) for p in self.splits[self.split]]
        if max_glosses is not None or max_samples_per_gloss is not None:
            self.paths = self._apply_limits(self.paths, max_glosses, max_samples_per_gloss)

        self.normalization = None
        if self.use_normalization:
            stats = self._load_json(self.data_dir / "normalization_stats.json")
            mean = np.asarray(stats["mean"], dtype=np.float32)
            std = np.asarray(stats["std"], dtype=np.float32)
            self.normalization = (mean, std)

    def _load_json(self, path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _apply_limits(self, paths: list[Path], max_glosses: int | None, max_samples_per_gloss: int | None) -> list[Path]:
        gloss_to_paths: dict[str, list[Path]] = {}
        for path in paths:
            gloss = self.path_to_gloss.get(str(path))
            if gloss is None:
                continue
            gloss_to_paths.setdefault(gloss, []).append(path)

        limited: list[Path] = []
        for gloss in sorted(gloss_to_paths):
            if max_glosses is not None and len(limited) >= max_glosses * (max_samples_per_gloss or len(gloss_to_paths[gloss])):
                break
            samples = gloss_to_paths[gloss]
            if max_samples_per_gloss is not None:
                samples = samples[:max_samples_per_gloss]
            limited.extend(samples)
            if max_glosses is not None and len(limited) >= max_glosses * (max_samples_per_gloss or len(samples)):
                break

        return limited

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        raw = load_pickle(path)
        sequence = extract_smplx_sequence(raw)
        sequence = adjust_sequence_length(sequence, self.seq_len, self.sequence_mode)
        if self.use_normalization and self.normalization is not None:
            sequence = normalize_sequence(sequence, *self.normalization)

        gloss = self._find_gloss_for_path(path)
        gloss_id = self.vocab.get(gloss, self.vocab.get("<UNK>"))

        return {
            "gloss": gloss,
            "gloss_id": torch.tensor(gloss_id, dtype=torch.long),
            "motion": torch.from_numpy(sequence).to(torch.float32),
            "path": str(path),
        }

    def _find_gloss_for_path(self, path: Path) -> str:
        gloss = self.path_to_gloss.get(str(path))
        if gloss is None:
            raise RuntimeError(f"Could not determine gloss for path: {path}")
        return gloss
