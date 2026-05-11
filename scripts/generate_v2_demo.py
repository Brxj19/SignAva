import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_v2.inference_v2 import generate_motion_v2
from src.model_v2.utils import OUTPUT_V2_DIR, load_vocab, safe_filename
from src.renderer.smplx_renderer import SMPLXRenderer


def _available_glosses() -> list[str]:
    vocab, _ = load_vocab()
    return sorted(gloss for gloss in vocab if gloss not in {"<PAD>", "<UNK>"})


def generate_v2_demo(
    glosses: list[str] | None = None,
    limit: int | None = None,
    render: bool = True,
    max_frames: int | None = None,
    decode_strategy: str = "greedy",
    temperature: float = 1.0,
    top_k: int = 0,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 0,
) -> list[dict]:
    selected = glosses or _available_glosses()
    if limit is not None:
        selected = selected[:limit]
    generated_dir = OUTPUT_V2_DIR / "generated"
    videos_dir = OUTPUT_V2_DIR / "videos"
    generated_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    renderer = SMPLXRenderer(force_flip_vertical=True, flip_vertical=False) if render else None
    results = []
    for gloss in selected:
        safe = safe_filename(gloss)
        motion_path = generated_dir / f"{safe}_v2_smplx.npy"
        video_path = videos_dir / f"{safe}_v2_animation.mp4"
        entry = {"gloss": gloss, "motion_path": str(motion_path), "video_path": str(video_path), "status": "pending"}
        try:
            _, metadata = generate_motion_v2(
                gloss,
                output_path=motion_path,
                greedy=decode_strategy == "greedy",
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            entry["metadata"] = metadata
            if renderer is not None:
                renderer.render_generated_motion(motion_path=motion_path, output_path=video_path, max_frames=max_frames)
            entry["status"] = "done"
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = str(exc)
        results.append(entry)

    index_path = OUTPUT_V2_DIR / "demo_index.json"
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate V2 motions and optional videos.")
    parser.add_argument("--gloss", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--decode-strategy", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    args = parser.parse_args()
    results = generate_v2_demo(
        args.gloss,
        args.limit,
        render=not args.no_render,
        max_frames=args.max_frames,
        decode_strategy=args.decode_strategy,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
