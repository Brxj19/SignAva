import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch

from src.data.dataset import SignAvatarDataset, denormalize_sequence
from src.model_v2.train_token_generator import _build_vqvae_from_checkpoint
from src.model_v2.utils import CHECKPOINT_V2_DIR, OUTPUT_V2_DIR, PROJECT_ROOT, safe_filename, select_device
from src.renderer.smplx_renderer import SMPLXRenderer


def _resolve_checkpoint(path: str | Path) -> Path:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"VQ-VAE checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _to_render_space(motion: np.ndarray, dataset: SignAvatarDataset) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float32)
    if dataset.use_normalization and dataset.normalization is not None:
        motion = denormalize_sequence(motion, *dataset.normalization)
    return motion.astype(np.float32)


def _sample_video_frames(video_path: Path, count: int = 12) -> tuple[list[np.ndarray], list[int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video has no readable frames: {video_path}")

    indices = np.linspace(0, total_frames - 1, num=min(count, total_frames), dtype=np.int32)
    frames: list[np.ndarray] = []
    used_indices: list[int] = []
    try:
        for frame_index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frames.append(frame)
            used_indices.append(int(frame_index))
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"Could not sample frames from video: {video_path}")
    return frames, used_indices


def _label_frame(frame: np.ndarray, label: str) -> np.ndarray:
    labeled = frame.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 28), (255, 255, 255), -1)
    cv2.putText(
        labeled,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return labeled


def _pad_frame(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if frame.shape[0] == target_h and frame.shape[1] == target_w:
        return frame
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)


def create_contact_sheet(
    original_video: Path,
    reconstructed_video: Path,
    output_path: Path,
    gloss: str,
    frames_per_row: int = 12,
) -> Path:
    original_frames, original_indices = _sample_video_frames(original_video, frames_per_row)
    reconstructed_frames, reconstructed_indices = _sample_video_frames(reconstructed_video, frames_per_row)
    frame_count = min(len(original_frames), len(reconstructed_frames), frames_per_row)
    if frame_count == 0:
        raise RuntimeError("No paired frames available for contact sheet")

    original_frames = original_frames[:frame_count]
    reconstructed_frames = reconstructed_frames[:frame_count]
    original_indices = original_indices[:frame_count]
    reconstructed_indices = reconstructed_indices[:frame_count]

    thumb_w = 160
    thumb_h = 160
    original_row = [
        _label_frame(_pad_frame(frame, thumb_h, thumb_w), f"Original f{frame_index}")
        for frame, frame_index in zip(original_frames, original_indices)
    ]
    reconstructed_row = [
        _label_frame(_pad_frame(frame, thumb_h, thumb_w), f"Recon f{frame_index}")
        for frame, frame_index in zip(reconstructed_frames, reconstructed_indices)
    ]

    top = np.concatenate(original_row, axis=1)
    bottom = np.concatenate(reconstructed_row, axis=1)
    title_h = 42
    title = np.full((title_h, top.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        title,
        f"V2 VQ-VAE Reconstruction Comparison: {gloss}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    sheet = np.concatenate([title, top, bottom], axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Failed to write contact sheet: {output_path}")
    return output_path


def _paths_for_sample(split: str, output_index: int, gloss: str) -> dict[str, Path]:
    safe_gloss = safe_filename(gloss)
    stem = f"sample_{output_index:03d}_gloss_{safe_gloss}"
    recon_root = OUTPUT_V2_DIR / "reconstructions" / split
    video_root = OUTPUT_V2_DIR / "videos" / "reconstructions" / split
    return {
        "original_npy": recon_root / "npy" / f"{stem}_original.npy",
        "reconstructed_npy": recon_root / "npy" / f"{stem}_reconstructed.npy",
        "metadata": recon_root / "metadata" / f"{stem}_metadata.json",
        "original_video": video_root / "original" / f"{stem}_original.mp4",
        "reconstructed_video": video_root / "reconstructed" / f"{stem}_reconstructed.mp4",
        "contact_sheet": video_root / "contact_sheets" / f"{stem}_comparison.jpg",
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def render_v2_reconstructions(
    checkpoint: str | Path = CHECKPOINT_V2_DIR / "vqvae_best.pth",
    split: str = "test",
    num_samples: int = 10,
    batch_size: int = 1,
    start_index: int = 0,
    fps: int = 20,
    no_render: bool = False,
) -> list[dict[str, Any]]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if fps <= 0:
        raise ValueError("fps must be positive")

    checkpoint_path = _resolve_checkpoint(checkpoint)
    device = select_device()
    vqvae = _build_vqvae_from_checkpoint(checkpoint_path, device)
    vqvae.eval()

    dataset = SignAvatarDataset(
        split=split,
        seq_len=vqvae.seq_len,
        sequence_mode="resample",
        canonicalize_camera=True,
        fix_betas=True,
    )
    if start_index >= len(dataset):
        raise IndexError(f"start_index={start_index} is outside split '{split}' with {len(dataset)} samples")

    end_index = min(start_index + num_samples, len(dataset))
    selected_indices = list(range(start_index, end_index))
    print(f"Rendering V2 reconstructions for {len(selected_indices)} {split} samples")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output root: {OUTPUT_V2_DIR / 'reconstructions' / split}")
    if batch_size != 1:
        print("Note: samples are rendered individually; --batch-size is accepted for CLI compatibility.")

    renderer = None if no_render else SMPLXRenderer(fps=fps)
    index_rows: list[dict[str, Any]] = []

    for output_index, dataset_index in enumerate(selected_indices):
        item = dataset[dataset_index]
        gloss = str(item["gloss"])
        gloss_id = int(item["gloss_id"].item())
        paths = _paths_for_sample(split, output_index, gloss)
        row = {
            "sample_index": dataset_index,
            "output_index": output_index,
            "gloss": gloss,
            "gloss_id": gloss_id,
            "original_npy": str(paths["original_npy"]),
            "reconstructed_npy": str(paths["reconstructed_npy"]),
            "original_video": str(paths["original_video"]) if not no_render else None,
            "reconstructed_video": str(paths["reconstructed_video"]) if not no_render else None,
            "contact_sheet": str(paths["contact_sheet"]) if not no_render else None,
            "status": "pending",
            "error": None,
        }

        print(f"[{output_index + 1}/{len(selected_indices)}] sample={dataset_index} gloss={gloss}")
        try:
            original_tensor = item["motion"].unsqueeze(0).to(device)
            with torch.no_grad():
                reconstructed_tensor, code_indices, vq_loss, perplexity = vqvae(original_tensor)

            original_motion = _to_render_space(item["motion"].detach().cpu().numpy(), dataset)
            reconstructed_motion = _to_render_space(
                reconstructed_tensor.squeeze(0).detach().cpu().numpy(),
                dataset,
            )

            if original_motion.shape != (vqvae.seq_len, vqvae.motion_dim):
                raise RuntimeError(f"Unexpected original motion shape: {original_motion.shape}")
            if reconstructed_motion.shape != (vqvae.seq_len, vqvae.motion_dim):
                raise RuntimeError(f"Unexpected reconstructed motion shape: {reconstructed_motion.shape}")

            paths["original_npy"].parent.mkdir(parents=True, exist_ok=True)
            paths["reconstructed_npy"].parent.mkdir(parents=True, exist_ok=True)
            np.save(paths["original_npy"], original_motion)
            np.save(paths["reconstructed_npy"], reconstructed_motion)

            metadata = {
                "sample_index": dataset_index,
                "output_index": output_index,
                "dataset_split": split,
                "gloss": gloss,
                "gloss_id": gloss_id,
                "original_path": item.get("path"),
                "checkpoint_path": str(checkpoint_path),
                "seq_len": int(vqvae.seq_len),
                "motion_dim": int(vqvae.motion_dim),
                "token_len": int(vqvae.token_len),
                "code_indices": code_indices.squeeze(0).detach().cpu().tolist(),
                "vq_loss": float(vq_loss.detach().cpu()),
                "perplexity": float(perplexity.detach().cpu()),
                "output_npy_paths": {
                    "original": str(paths["original_npy"]),
                    "reconstructed": str(paths["reconstructed_npy"]),
                },
                "output_video_paths": {
                    "original": str(paths["original_video"]) if not no_render else None,
                    "reconstructed": str(paths["reconstructed_video"]) if not no_render else None,
                    "contact_sheet": str(paths["contact_sheet"]) if not no_render else None,
                },
            }

            if renderer is not None:
                renderer.render_generated_motion(paths["original_npy"], paths["original_video"])
                renderer.render_generated_motion(paths["reconstructed_npy"], paths["reconstructed_video"])
                try:
                    create_contact_sheet(
                        paths["original_video"],
                        paths["reconstructed_video"],
                        paths["contact_sheet"],
                        gloss,
                    )
                except Exception as exc:
                    print(f"Warning: failed to create contact sheet for sample {dataset_index}: {exc}")
                    metadata["contact_sheet_error"] = str(exc)

            _write_json(paths["metadata"], metadata)
            row["status"] = "done"
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            print(f"Failed sample {dataset_index} ({gloss}): {exc}")

        index_rows.append(row)

    index_path = OUTPUT_V2_DIR / "reconstructions" / split / "reconstruction_index.json"
    _write_json(index_path, index_rows)
    completed = sum(1 for row in index_rows if row["status"] == "done")
    failed = sum(1 for row in index_rows if row["status"] == "failed")
    print("\nV2 reconstruction rendering complete")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Index: {index_path}")
    print(f"Output directory: {OUTPUT_V2_DIR / 'reconstructions' / split}")
    return index_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Render V2 MotionVQVAE original vs reconstructed comparisons.")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_V2_DIR / "vqvae_best.pth"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    render_v2_reconstructions(
        checkpoint=args.checkpoint,
        split=args.split,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        start_index=args.start_index,
        fps=args.fps,
        no_render=args.no_render,
    )


if __name__ == "__main__":
    main()
