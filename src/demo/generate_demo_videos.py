import argparse
import json
import re
from pathlib import Path
from typing import Any

from src.config import OUTPUT_DIR
from src.model.inference import generate_motion, list_available_glosses
from src.renderer.smplx_renderer import SMPLXRenderer


GENERATED_DIR = OUTPUT_DIR / "generated"
VIDEOS_DIR = OUTPUT_DIR / "videos"
DEMO_DIR = OUTPUT_DIR / "demo"
DEMO_INDEX_PATH = DEMO_DIR / "demo_index.json"


def safe_filename(gloss: str) -> str:
    safe = gloss.lower().strip().replace(" ", "_")
    safe = re.sub(r"[^a-z0-9_-]+", "", safe)
    safe = re.sub(r"_+", "_", safe).strip("_")
    if not safe:
        safe = "gloss"
    return safe


def _select_glosses(glosses: list[str] | None, limit: int | None) -> list[str]:
    available = list_available_glosses()
    print(f"Available glosses: {len(available)}")

    if glosses:
        available_set = set(available)
        missing = [gloss for gloss in glosses if gloss not in available_set]
        if missing:
            raise ValueError(f"Requested glosses are not available: {missing}.")
        selected = glosses
    else:
        selected = available

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive when provided.")
        selected = selected[:limit]

    return selected


def _paths_for_gloss(gloss: str) -> tuple[str, Path, Path, Path]:
    safe_gloss = safe_filename(gloss)
    motion_path = GENERATED_DIR / f"{safe_gloss}_smplx.npy"
    metadata_path = GENERATED_DIR / f"{safe_gloss}_metadata.json"
    video_path = VIDEOS_DIR / f"{safe_gloss}_animation.mp4"
    return safe_gloss, motion_path, metadata_path, video_path


def _write_demo_index(results: list[dict[str, Any]]) -> Path:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    with DEMO_INDEX_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return DEMO_INDEX_PATH


def generate_demo_videos(
    glosses: list[str] | None = None,
    limit: int | None = None,
    skip_existing: bool = False,
    max_frames: int | None = None,
    force_flip_vertical: bool = True,
    view_yaw_degrees: float = 0.0,
    view_pitch_degrees: float = 0.0,
    view_roll_degrees: float = 0.0,
) -> list[dict[str, Any]]:
    selected_glosses = _select_glosses(glosses, limit)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    renderer: SMPLXRenderer | None = None

    for gloss in selected_glosses:
        safe_gloss, motion_path, metadata_path, video_path = _paths_for_gloss(gloss)
        entry: dict[str, Any] = {
            "gloss": gloss,
            "safe_gloss": safe_gloss,
            "generated_motion_path": str(motion_path),
            "metadata_path": str(metadata_path),
            "video_path": str(video_path),
            "status": "pending",
            "error": None,
        }

        print(f"Generating: {gloss}")

        try:
            if skip_existing and video_path.exists():
                entry["status"] = "skipped"
                print(f"Skipping existing video: {video_path}")
                results.append(entry)
                continue

            _, metadata = generate_motion(gloss, output_path=motion_path)
            metadata["demo_motion_path"] = str(motion_path)
            metadata["demo_video_path"] = str(video_path)
            with metadata_path.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            print(f"Motion saved: {motion_path}")
            print(f"Metadata saved: {metadata_path}")

            if renderer is None:
                renderer = SMPLXRenderer(
                    force_flip_vertical=force_flip_vertical,
                    flip_vertical=False,
                    view_yaw_degrees=view_yaw_degrees,
                    view_pitch_degrees=view_pitch_degrees,
                    view_roll_degrees=view_roll_degrees,
                )

            rendered_path = renderer.render_generated_motion(
                motion_path=motion_path,
                output_path=video_path,
                max_frames=max_frames,
            )
            print(f"Video saved: {rendered_path}")

            entry["status"] = "done"
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = str(exc)
            print(f"Failed: {gloss}: {exc}")

        results.append(entry)

    index_path = _write_demo_index(results)
    print(f"Done: {len(results)} entries")
    print(f"Demo index saved: {index_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo videos for available glosses.")
    parser.add_argument("--gloss", action="append", default=None, help="Gloss to render. Can be repeated.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of glosses to render.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip glosses whose output video already exists.")
    parser.add_argument("--max-frames", type=int, default=None, help="Render at most this many frames per video.")
    parser.add_argument("--force-flip-vertical", dest="force_flip_vertical", action="store_true", default=True)
    parser.add_argument("--no-force-flip-vertical", dest="force_flip_vertical", action="store_false")
    parser.add_argument("--view-yaw", type=float, default=0.0)
    parser.add_argument("--view-pitch", type=float, default=0.0)
    parser.add_argument("--view-roll", type=float, default=0.0)
    args = parser.parse_args()

    results = generate_demo_videos(
        glosses=args.gloss,
        limit=args.limit,
        skip_existing=args.skip_existing,
        max_frames=args.max_frames,
        force_flip_vertical=args.force_flip_vertical,
        view_yaw_degrees=args.view_yaw,
        view_pitch_degrees=args.view_pitch,
        view_roll_degrees=args.view_roll,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
