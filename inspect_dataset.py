# ============================================================
# Inspect SignAvatar PKL dataset shapes and sequence lengths
# ============================================================

from pathlib import Path
import os
import sys
import json
import pickle
import io
import csv
from collections import Counter, defaultdict

import numpy as np
import torch

# ------------------------------------------------------------
# 1. Project setup
# ------------------------------------------------------------
PROJECT_ROOT = Path("/home/image/Desktop/SignAvatar_Generative_Project")

# For local Linux lab PC, uncomment this and comment the Colab path above:
# PROJECT_ROOT = Path.home() / "Desktop/SignAvatar_Generative_Project"

os.chdir(PROJECT_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("Project root:", PROJECT_ROOT)
print("Current directory:", Path.cwd())

# ------------------------------------------------------------
# 2. Dataset paths
# ------------------------------------------------------------
PKL_DIR = PROJECT_ROOT / "data/annotations/wlasl_pkls_cropFalse_defult_shape"
INDEX_PATH_CANDIDATES = [
    PROJECT_ROOT / "data/processed/index.json",
    PROJECT_ROOT / "data/index.json",
]

OUTPUT_DIR = PROJECT_ROOT / "outputs/inspection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("PKL dir exists:", PKL_DIR.exists())
print("PKL dir:", PKL_DIR)

# ------------------------------------------------------------
# 3. CPU-safe pickle loader
# ------------------------------------------------------------
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(
                io.BytesIO(b),
                map_location=torch.device("cpu"),
                weights_only=False
            )
        return super().find_class(module, name)

def load_pickle_cpu(path):
    with open(path, "rb") as f:
        return CPU_Unpickler(f).load()

# ------------------------------------------------------------
# 4. Load index.json if available
# ------------------------------------------------------------
index_path = None
for p in INDEX_PATH_CANDIDATES:
    if p.exists():
        index_path = p
        break

video_to_gloss = {}
gloss_to_paths = {}

if index_path is not None:
    print("Using index:", index_path)
    with open(index_path, "r") as f:
        index = json.load(f)

    video_to_gloss = index.get("video_to_gloss", {})
    gloss_to_paths = index.get("gloss_to_paths", {})
else:
    print("No index.json found. Will inspect PKLs by glob only.")

# Build reverse lookup: file name/path -> gloss
path_to_gloss = {}

for gloss, paths in gloss_to_paths.items():
    for p in paths:
        p_obj = Path(p)
        path_to_gloss[str(p)] = gloss
        path_to_gloss[p_obj.name] = gloss

# ------------------------------------------------------------
# 5. Collect PKL files
# ------------------------------------------------------------
pkl_files = sorted(PKL_DIR.glob("*.pkl"))

print("Total PKL files found:", len(pkl_files))

if len(pkl_files) == 0:
    raise FileNotFoundError(f"No .pkl files found in {PKL_DIR}")

# ------------------------------------------------------------
# 6. Inspect files
# ------------------------------------------------------------
rows = []
errors = []

smplx_shapes = Counter()
seq_lengths = []
motion_dims = Counter()
key_counter = Counter()
gloss_counter = Counter()
gloss_lengths = defaultdict(list)

sample_key_shapes = {}

for i, pkl_path in enumerate(pkl_files, start=1):
    if i % 100 == 0:
        print(f"Scanned {i}/{len(pkl_files)} files...")

    try:
        data = load_pickle_cpu(pkl_path)

        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data)}")

        keys = list(data.keys())
        for k in keys:
            key_counter[k] += 1

        # detect gloss
        gloss = path_to_gloss.get(pkl_path.name, None)

        if gloss is None:
            # Try relative path match
            try:
                rel = str(pkl_path.relative_to(PROJECT_ROOT))
                gloss = path_to_gloss.get(rel, "UNKNOWN")
            except Exception:
                gloss = "UNKNOWN"

        gloss_counter[gloss] += 1

        # inspect all key shapes for first few examples
        for k, v in data.items():
            if k not in sample_key_shapes:
                try:
                    if hasattr(v, "shape"):
                        sample_key_shapes[k] = str(tuple(v.shape))
                    elif isinstance(v, list):
                        sample_key_shapes[k] = f"list_len_{len(v)}"
                    else:
                        sample_key_shapes[k] = str(type(v))
                except Exception:
                    sample_key_shapes[k] = "unknown"

        if "smplx" not in data:
            raise KeyError("Missing key 'smplx'")

        smplx = np.asarray(data["smplx"])

        if smplx.ndim != 2:
            raise ValueError(f"smplx is not 2D, shape={smplx.shape}")

        T, D = smplx.shape

        smplx_shapes[str(tuple(smplx.shape))] += 1
        seq_lengths.append(T)
        motion_dims[D] += 1
        gloss_lengths[gloss].append(T)

        rows.append({
            "file": str(pkl_path),
            "file_name": pkl_path.name,
            "gloss": gloss,
            "smplx_shape": str(tuple(smplx.shape)),
            "seq_len_T": T,
            "motion_dim_D": D,
            "duration_at_20fps_sec": round(T / 20.0, 3),
            "keys": "|".join(keys),
        })

    except Exception as e:
        errors.append({
            "file": str(pkl_path),
            "error": str(e),
        })

# ------------------------------------------------------------
# 7. Summary stats
# ------------------------------------------------------------
seq_arr = np.array(seq_lengths, dtype=np.float32)

summary = {
    "total_pkl_files": len(pkl_files),
    "successfully_scanned": len(rows),
    "failed_files": len(errors),

    "unique_smplx_shapes_count": len(smplx_shapes),
    "most_common_smplx_shapes": smplx_shapes.most_common(20),

    "motion_dim_distribution": motion_dims.most_common(),

    "sequence_length_stats": {
        "min": int(seq_arr.min()) if len(seq_arr) else None,
        "max": int(seq_arr.max()) if len(seq_arr) else None,
        "mean": float(seq_arr.mean()) if len(seq_arr) else None,
        "median": float(np.median(seq_arr)) if len(seq_arr) else None,
        "std": float(seq_arr.std()) if len(seq_arr) else None,
        "p10": float(np.percentile(seq_arr, 10)) if len(seq_arr) else None,
        "p25": float(np.percentile(seq_arr, 25)) if len(seq_arr) else None,
        "p75": float(np.percentile(seq_arr, 75)) if len(seq_arr) else None,
        "p90": float(np.percentile(seq_arr, 90)) if len(seq_arr) else None,
    },

    "duration_at_20fps_stats_sec": {
        "min": float(seq_arr.min() / 20.0) if len(seq_arr) else None,
        "max": float(seq_arr.max() / 20.0) if len(seq_arr) else None,
        "mean": float(seq_arr.mean() / 20.0) if len(seq_arr) else None,
        "median": float(np.median(seq_arr) / 20.0) if len(seq_arr) else None,
    },

    "key_frequency": key_counter.most_common(),

    "sample_key_shapes": sample_key_shapes,

    "gloss_count": len(gloss_counter),
    "top_glosses_by_sample_count": gloss_counter.most_common(30),
}

# Per-gloss stats
per_gloss_stats = []

for gloss, lengths in sorted(gloss_lengths.items()):
    arr = np.array(lengths, dtype=np.float32)

    per_gloss_stats.append({
        "gloss": gloss,
        "num_samples": len(lengths),
        "min_T": int(arr.min()),
        "max_T": int(arr.max()),
        "mean_T": float(arr.mean()),
        "median_T": float(np.median(arr)),
        "mean_duration_at_20fps_sec": float(arr.mean() / 20.0),
    })

summary["per_gloss_stats"] = per_gloss_stats

# ------------------------------------------------------------
# 8. Save reports
# ------------------------------------------------------------
summary_path = OUTPUT_DIR / "dataset_shape_summary.json"
csv_path = OUTPUT_DIR / "pkl_shape_report.csv"
errors_path = OUTPUT_DIR / "pkl_scan_errors.json"
per_gloss_csv_path = OUTPUT_DIR / "per_gloss_length_stats.csv"

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

with open(errors_path, "w") as f:
    json.dump(errors, f, indent=2)

with open(csv_path, "w", newline="") as f:
    fieldnames = [
        "file",
        "file_name",
        "gloss",
        "smplx_shape",
        "seq_len_T",
        "motion_dim_D",
        "duration_at_20fps_sec",
        "keys",
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

with open(per_gloss_csv_path, "w", newline="") as f:
    fieldnames = [
        "gloss",
        "num_samples",
        "min_T",
        "max_T",
        "mean_T",
        "median_T",
        "mean_duration_at_20fps_sec",
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(per_gloss_stats)

# ------------------------------------------------------------
# 9. Print useful summary
# ------------------------------------------------------------
print("\n================ DATASET SHAPE SUMMARY ================")

print("Total PKLs:", summary["total_pkl_files"])
print("Scanned successfully:", summary["successfully_scanned"])
print("Failed files:", summary["failed_files"])

print("\nMotion dimension distribution:")
for dim, count in motion_dims.most_common():
    print(f"  D={dim}: {count} files")

print("\nMost common SMPL-X shapes:")
for shape, count in smplx_shapes.most_common(15):
    print(f"  {shape}: {count}")

print("\nSequence length stats:")
for k, v in summary["sequence_length_stats"].items():
    print(f"  {k}: {v}")

print("\nDuration stats at 20 FPS:")
for k, v in summary["duration_at_20fps_stats_sec"].items():
    print(f"  {k}: {v}")

print("\nDataset keys and frequency:")
for key, count in key_counter.most_common():
    print(f"  {key}: {count}")

print("\nSample key shapes:")
for key, shape in sample_key_shapes.items():
    print(f"  {key}: {shape}")

print("\nTop glosses by sample count:")
for gloss, count in gloss_counter.most_common(20):
    print(f"  {gloss}: {count}")

print("\nSaved reports:")
print("  Summary:", summary_path)
print("  CSV:", csv_path)
print("  Errors:", errors_path)
print("  Per-gloss CSV:", per_gloss_csv_path)