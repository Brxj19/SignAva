import zipfile
from pathlib import Path

from src.config import ANNOTATION_DIR, RAW_DIR


def find_zip_file() -> Path | None:
    zip_files = sorted(RAW_DIR.glob("*.zip"))
    if not zip_files:
        return None
    if len(zip_files) > 1:
        print(
            "Warning: multiple zip files found in data/raw. Using the first one: "
            f"{zip_files[0].name}"
        )
    return zip_files[0]


def count_pkl_files() -> int:
    return sum(1 for _ in ANNOTATION_DIR.rglob("*.pkl"))


def _is_safe_path(base_dir: Path, target_path: Path) -> bool:
    try:
        base_dir = base_dir.resolve()
        target_path = target_path.resolve()
        return str(target_path).startswith(str(base_dir))
    except RuntimeError:
        return False


def extract_dataset() -> None:
    existing_count = count_pkl_files()
    if existing_count > 0:
        print(
            f"Found {existing_count} existing .pkl file(s) in {ANNOTATION_DIR}. Skipping extraction."
        )
        return

    zip_path = find_zip_file()
    if zip_path is None:
        raise FileNotFoundError(
            "No zip archive found in data/raw/. Place the dataset zip file there before running extraction."
        )

    print(f"Extracting archive: {zip_path.name} -> {ANNOTATION_DIR}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = [name for name in archive.namelist() if name.endswith(".pkl")]
        if not members:
            print("No .pkl files found inside the archive. Extracting all archive contents.")
            members = archive.namelist()

        for member in members:
            destination = ANNOTATION_DIR / member
            if not _is_safe_path(ANNOTATION_DIR, destination):
                print(f"Skipping unsafe archive member: {member}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, open(destination, "wb") as target:
                target.write(source.read())

    final_count = count_pkl_files()
    print(f"Extraction complete. Found {final_count} .pkl file(s) in {ANNOTATION_DIR}.")


if __name__ == "__main__":
    extract_dataset()
