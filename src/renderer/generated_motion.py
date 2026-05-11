import pickle
from pathlib import Path

import numpy as np

from src.config import OUTPUT_DIR


DEFAULT_GENERATED_MOTION_PATH = OUTPUT_DIR / "generated" / "generated_smplx.npy"
DEFAULT_DEBUG_PKL_PATH = OUTPUT_DIR / "generated" / "generated_smplx_debug.pkl"


def load_generated_motion(path: str | Path | None = None) -> np.ndarray:
    motion_path = Path(path) if path is not None else DEFAULT_GENERATED_MOTION_PATH
    if not motion_path.exists():
        raise FileNotFoundError(
            f"Generated motion file not found: {motion_path}. "
            "Run generate_motion() or python3 -m src.model.inference --text \"about\" first."
        )

    motion = np.load(motion_path)
    if motion.ndim != 2 or motion.shape[1] != 182:
        raise ValueError(f"Expected generated motion shape (T, 182), got {motion.shape} from {motion_path}.")

    return motion.astype(np.float32)


def save_generated_motion_as_pkl_like(
    motion: np.ndarray,
    output_path: str | Path | None = None,
) -> Path:
    motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] != 182:
        raise ValueError(f"Expected motion shape (T, 182), got {motion.shape}.")

    debug_path = Path(output_path) if output_path is not None else DEFAULT_DEBUG_PKL_PATH
    debug_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "smplx": motion,
        "generated": True,
        "source": "model_inference",
    }
    with debug_path.open("wb") as f:
        pickle.dump(payload, f)

    return debug_path
