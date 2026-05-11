import json
import re
from pathlib import Path
from typing import Any

import torch

from src.config import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT


CHECKPOINT_V2_DIR = PROJECT_ROOT / "checkpoints_v2"
OUTPUT_V2_DIR = OUTPUT_DIR / "v2"


def select_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_vocab() -> tuple[dict[str, int], dict[int, str]]:
    vocab = load_json(DATA_DIR / "vocab.json")
    vocab = {str(gloss): int(gloss_id) for gloss, gloss_id in vocab.items()}
    return vocab, {gloss_id: gloss for gloss, gloss_id in vocab.items()}


def safe_torch_load(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def safe_filename(text: str) -> str:
    safe = text.lower().strip().replace(" ", "_")
    safe = re.sub(r"[^a-z0-9_-]+", "", safe)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "gloss"


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
