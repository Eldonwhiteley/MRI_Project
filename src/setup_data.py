"""setup_data.py
Helper utility to automatically detect and extract train-shifted and valid-shifted
tar archives (.tar.gz / .tar) into the gitignored `data/` directory.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tarfile
from glob import glob
from pathlib import Path
from typing import List, Optional


def find_archive(name_pattern: str, search_dirs: List[Path]) -> Optional[Path]:
    """Search given directories for an archive matching the pattern."""
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        matches = list(directory.glob(name_pattern))
        if matches:
            # Pick first matching non-hidden file
            for match in matches:
                if not match.name.startswith("._"):
                    return match
    return None


def clean_appledouble(target_dir: Path) -> int:
    """Remove macOS AppleDouble metadata files (._*) that corrupt loaders."""
    count = 0
    for root, _, files in os.walk(target_dir):
        for f in files:
            if f.startswith("._"):
                try:
                    os.remove(os.path.join(root, f))
                    count += 1
                except OSError:
                    pass
    return count


def extract_archives(
    archive_dir: Optional[str] = None,
    dest_dir: str = "data",
    clean_mac_meta: bool = True,
) -> bool:
    """Extract train-shifted and valid-shifted archives into dest_dir.

    Args:
        archive_dir: Optional directory where .tar.gz archives are stored.
        dest_dir: Target directory where data should be extracted (default: 'data').
        clean_mac_meta: Whether to remove macOS ._* files after extraction.

    Returns:
        True if datasets are present and ready, False otherwise.
    """
    dest_path = Path(dest_dir).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)

    search_dirs = [
        dest_path,
        Path(".").resolve(),
        Path("..").resolve(),
        Path.home() / "Downloads",
    ]
    if archive_dir:
        search_dirs.insert(0, Path(archive_dir).resolve())

    targets = [
        ("train-shifted", "train-shifted*.tar*"),
        ("valid-shifted", "valid-shifted*.tar*"),
    ]

    for folder_name, pattern in targets:
        target_folder = dest_path / folder_name
        # Check if folder already contains .im files
        existing_im = list(target_folder.glob("**/*.im")) if target_folder.is_dir() else []
        # Filter out ._
        existing_im = [f for f in existing_im if not f.name.startswith("._")]

        if existing_im:
            print(f"[{folder_name}] Already extracted: found {len(existing_im)} .im files in {target_folder}")
            continue

        archive_file = find_archive(pattern, search_dirs)
        if archive_file is None:
            print(f"[{folder_name}] WARNING: Could not find archive matching '{pattern}' in:")
            for d in search_dirs:
                print(f"    - {d}")
            continue

        print(f"[{folder_name}] Extracting {archive_file.name} to {dest_path}...")
        try:
            with tarfile.open(archive_file, "r:*") as tar:
                tar.extractall(path=dest_path)
            print(f"[{folder_name}] Extraction complete!")
        except Exception as e:
            print(f"[{folder_name}] Error extracting {archive_file}: {e}")

        # Flatten any redundant nesting (e.g. data/train-shifted/train-shifted -> data/train-shifted)
        nested = target_folder / folder_name
        if nested.is_dir():
            for item in nested.iterdir():
                dest_item = target_folder / item.name
                if not dest_item.exists():
                    shutil.move(str(item), str(target_folder))
            try:
                nested.rmdir()
            except OSError:
                pass

        if clean_mac_meta and target_folder.is_dir():
            cleaned = clean_appledouble(target_folder)
            if cleaned > 0:
                print(f"[{folder_name}] Cleaned {cleaned} macOS AppleDouble metadata files.")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MRI dataset archives into data/ folder.")
    parser.add_argument(
        "--archive-dir",
        type=str,
        default=None,
        help="Directory containing train-shifted.tar.gz and valid-shifted.tar.gz (default: search Downloads, project root, and data/)",
    )
    parser.add_argument(
        "--dest-dir",
        type=str,
        default="data",
        help="Destination directory for extracted dataset (default: 'data')",
    )
    args = parser.parse_args()

    extract_archives(archive_dir=args.archive_dir, dest_dir=args.dest_dir)
