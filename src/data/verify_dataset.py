import json
import pickle
from pathlib import Path

import numpy as np
import torch

from src.config import ANNOTATION_DIR, PROCESSED_DIR


REPORT_PATH = PROCESSED_DIR / "dataset_verification_report.json"


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def describe_object(obj):
    result = {
        "type": type(obj).__name__,
        "keys": [],
    }
    if isinstance(obj, dict):
        result["keys"] = sorted(obj.keys())
    elif hasattr(obj, "__dict__"):
        result["keys"] = sorted(vars(obj).keys())
    return result


def normalize_sequence(sequence):
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.detach().cpu().numpy()
    if isinstance(sequence, np.ndarray):
        return sequence
    if isinstance(sequence, list):
        try:
            return np.asarray(sequence)
        except Exception:
            return None
    return None


def inspect_file(path: Path) -> dict:
    result = {
        "path": str(path.relative_to(Path.cwd())),
        "smplx_present": False,
        "smplx_shape": None,
        "object": None,
        "error": None,
    }

    try:
        data = load_pickle(path)
        result["object"] = describe_object(data)

        candidate = None
        if isinstance(data, dict) and "smplx" in data:
            candidate = data["smplx"]
        elif hasattr(data, "smplx"):
            candidate = getattr(data, "smplx")

        if candidate is not None:
            array = normalize_sequence(candidate)
            if array is not None:
                result["smplx_present"] = True
                result["smplx_shape"] = list(array.shape)
            else:
                result["error"] = "smplx value could not be converted to array"
        else:
            result["error"] = "smplx key not found"
    except Exception as exc:
        result["error"] = f"Failed to load file: {exc!r}"

    return result


def build_report():
    pkl_files = sorted(ANNOTATION_DIR.rglob("*.pkl"))
    total_files = len(pkl_files)
    sample_count = min(5, total_files)
    sampled_files = pkl_files[:sample_count]

    verification_results = [inspect_file(path) for path in sampled_files]
    files_with_smplx = sum(1 for item in verification_results if item["smplx_present"])
    files_without_smplx = sample_count - files_with_smplx

    report = {
        "total_pkl_files": total_files,
        "sampled_files": [str(path.relative_to(Path.cwd())) for path in sampled_files],
        "verification": verification_results,
        "summary": {
            "sampled_files": sample_count,
            "files_with_smplx": files_with_smplx,
            "files_without_smplx": files_without_smplx,
        },
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Scanned {total_files} .pkl file(s) under {ANNOTATION_DIR}.")
    print(f"Verified {sample_count} sample file(s).")
    if files_with_smplx:
        for entry in verification_results:
            if entry["smplx_present"]:
                print(
                    f"{entry['path']}: smplx shape = {entry['smplx_shape']}"
                )
            else:
                print(f"{entry['path']}: {entry['error']}")
    else:
        print("No sample files contained a valid smplx entry.")
    print(f"Saved verification report to {REPORT_PATH}")


if __name__ == "__main__":
    build_report()


# Alias for notebook-friendly import
def verify_dataset(*args, **kwargs):
    """
    Notebook-friendly wrapper around build_report().
    This allows:
        from src.data.verify_dataset import verify_dataset
    """
    return build_report(*args, **kwargs)
