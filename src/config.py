from pathlib import Path
from typing import Optional


def find_project_root(start_path: Optional[Path] = None) -> Path:
    if start_path is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start_path)
    start_path = start_path.resolve()

    for path in [start_path] + list(start_path.parents):
        if (path / "data").is_dir() and (path / "human_models").is_dir() and (path / "notebooks").is_dir():
            return path

    raise RuntimeError(
        "Could not find project root. Looking for directories: data/, human_models/, notebooks/. "
        "Run this command from anywhere inside the project tree."
    )


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ANNOTATION_DIR = DATA_DIR / "annotations"
PROCESSED_DIR = DATA_DIR / "processed"
HUMAN_MODELS_DIR = PROJECT_ROOT / "human_models"
SMPLX_MODEL_DIR = HUMAN_MODELS_DIR / "smplx"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

for path in (ANNOTATION_DIR, PROCESSED_DIR, OUTPUT_DIR, CHECKPOINT_DIR):
    path.mkdir(parents=True, exist_ok=True)
