import shutil
from datetime import datetime
from pathlib import Path

from src.config import CHECKPOINT_DIR, PROJECT_ROOT


CHECKPOINT_EXTENSIONS = {".pth", ".pt", ".ckpt"}


def find_checkpoint_files(checkpoint_dir: Path = CHECKPOINT_DIR) -> list[Path]:
    if not checkpoint_dir.exists():
        return []
    return sorted(
        path
        for path in checkpoint_dir.iterdir()
        if path.is_file() and path.suffix.lower() in CHECKPOINT_EXTENSIONS
    )


def backup_and_clean_checkpoints(
    checkpoint_dir: Path = CHECKPOINT_DIR,
    backups_root: Path | None = None,
) -> dict:
    backups_root = backups_root or PROJECT_ROOT / "backups"
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
    backup_dir = backups_root / f"checkpoints_before_weighted_loss_{timestamp}"
    checkpoint_files = find_checkpoint_files(checkpoint_dir)

    backup_dir.mkdir(parents=True, exist_ok=False)

    copied_files: list[Path] = []
    for source_path in checkpoint_files:
        destination_path = backup_dir / source_path.name
        shutil.copy2(source_path, destination_path)
        copied_files.append(destination_path)

    missing_copies = [path.name for path in checkpoint_files if not (backup_dir / path.name).exists()]
    if missing_copies:
        raise RuntimeError(f"Backup incomplete. Missing copied files: {missing_copies}")

    removed_files: list[Path] = []
    for source_path in checkpoint_files:
        source_path.unlink()
        removed_files.append(source_path)

    return {
        "backup_dir": backup_dir,
        "backed_up_count": len(copied_files),
        "removed_count": len(removed_files),
        "backed_up_files": copied_files,
    }


def main() -> None:
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    checkpoint_files = find_checkpoint_files(CHECKPOINT_DIR)
    print(f"Checkpoint files found: {len(checkpoint_files)}")

    result = backup_and_clean_checkpoints()

    print("\nCheckpoint backup complete.")
    print(f"Number of checkpoint files backed up: {result['backed_up_count']}")
    print(f"Backup folder path: {result['backup_dir']}")
    print(f"Checkpoint files removed from checkpoints/: {result['removed_count']}")
    print("checkpoints/ folder cleaned; .gitkeep preserved if present.")

    if result["backed_up_files"]:
        print("\nBacked up files:")
        for path in result["backed_up_files"]:
            print(f"- {path.name}")
    else:
        print("\nNo checkpoint files were present; created an empty timestamped backup folder.")


if __name__ == "__main__":
    main()
