"""
Delete the augmented-image JPEGs from one or more training runs to
reclaim disk space. The manifest CSV (~MB) is kept so the row count,
filenames, and labels are still reproducible alongside the run's
config.yaml + aug_seed.

By default operates on every runs/run_NNN/augmented/ under
training/corner/runs/. Pass one or more run dirs to target specific
runs only.

Usage:
    uv run scripts/clean_augmented.py                          # all runs
    uv run scripts/clean_augmented.py --dry-run                # preview
    uv run scripts/clean_augmented.py training/corner/runs/run_034
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from augment_corners import image_dir_for  # noqa: E402


RUNS_DIR = Path("training/corner/runs")


def dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:7.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def collect_run_dirs(args_run_dirs: list[Path]) -> list[Path]:
    if args_run_dirs:
        return args_run_dirs
    if not RUNS_DIR.is_dir():
        return []
    return sorted(p for p in RUNS_DIR.iterdir()
                  if p.is_dir() and p.name.startswith("run_"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("run_dirs", nargs="*", type=Path,
                        help="Specific run directories. If omitted, "
                             f"all runs under {RUNS_DIR} are scanned.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be deleted, change nothing.")
    args = parser.parse_args()

    targets = collect_run_dirs(args.run_dirs)
    if not targets:
        print(f"No run directories found under {RUNS_DIR}")
        sys.exit(0)

    label = "Would delete" if args.dry_run else "Deleting"
    total_freed = 0
    n_deleted = 0
    n_skipped = 0

    for run_dir in targets:
        aug = run_dir / "augmented"
        img_dir = image_dir_for(aug)
        if not img_dir.is_dir():
            n_skipped += 1
            print(f"  {'skip':<14} {run_dir.name}: "
                  f"no {img_dir.relative_to(run_dir)} dir")
            continue
        size = dir_size_bytes(img_dir)
        total_freed += size
        n_deleted += 1
        print(f"  {label.lower():<14} {run_dir.name}: "
              f"{img_dir.relative_to(run_dir)}  {format_bytes(size)}")
        if not args.dry_run:
            shutil.rmtree(img_dir)

    print()
    print(f"{label} {n_deleted} dir(s), skipped {n_skipped}, "
          f"total {format_bytes(total_freed)}")
    if args.dry_run and n_deleted:
        print("(re-run without --dry-run to actually delete)")


if __name__ == "__main__":
    main()
