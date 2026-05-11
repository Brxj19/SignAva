import json
from collections import Counter
from pathlib import Path

from src.config import ANNOTATION_DIR, PROCESSED_DIR, RAW_DIR

INDEX_PATH = PROCESSED_DIR / "index.json"
WLASL_JSON = RAW_DIR / "WLASL_v0.3.json"


def load_wlasl_mapping() -> dict[str, str]:
    if not WLASL_JSON.exists():
        print(f"No WLASL metadata found at {WLASL_JSON}. Gloss mapping will be empty.")
        return {}

    with WLASL_JSON.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    mapping: dict[str, str] = {}
    if isinstance(data, dict):
        entries = data.get("data") or data.get("entries") or []
    else:
        entries = data

    if not isinstance(entries, list):
        print("Unexpected JSON format for WLASL metadata. Expected a list of gloss entries.")
        return mapping

    for item in entries:
        if not isinstance(item, dict):
            continue
        gloss = item.get("gloss")
        instances = item.get("instances") or []
        if not gloss or not isinstance(instances, list):
            continue
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            video_id = instance.get("video_id") or instance.get("id")
            if video_id:
                mapping[str(video_id)] = gloss
    return mapping


def build_index() -> None:
    video_to_gloss = load_wlasl_mapping()
    pkl_paths = sorted(ANNOTATION_DIR.rglob("*.pkl"))

    matched = 0
    matched_glosses: dict[str, list[str]] = {}
    unmatched = []

    for pkl_path in pkl_paths:
        video_id = pkl_path.stem
        gloss = video_to_gloss.get(video_id)
        if gloss:
            matched += 1
            matched_glosses.setdefault(gloss, []).append(str(pkl_path.relative_to(Path.cwd())))
        else:
            unmatched.append(str(pkl_path.relative_to(Path.cwd())))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    index = {
        "video_to_gloss": video_to_gloss,
        "gloss_to_paths": matched_glosses,
        "matched_pkl_files": matched,
        "unmatched_pkl_files": unmatched,
        "total_pkl_files": len(pkl_paths),
    }

    with INDEX_PATH.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2)

    gloss_counts = Counter({gloss: len(paths) for gloss, paths in matched_glosses.items()})
    print(f"Total .pkl files: {len(pkl_paths)}")
    print(f"Matched .pkl files: {matched}")
    print(f"Unmatched .pkl files: {len(unmatched)}")
    print(f"Total glosses in WLASL metadata: {len(video_to_gloss)}")
    print("Top 20 glosses by sample count:")
    for gloss, count in gloss_counts.most_common(20):
        print(f"  {gloss}: {count}")
    print(f"Saved index to {INDEX_PATH}")


if __name__ == "__main__":
    build_index()
