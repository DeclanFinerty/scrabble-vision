"""
Run a sequence of training configs back-to-back, with prediction
visualization and a metrics summary after each. Designed for
overnight batches: a single bad config doesn't poison the rest --
its failure is recorded and the next config still runs.

For each config:
  1. uv run scripts/train_corners.py --config <yaml>
  2. uv run scripts/visualize_predictions.py <new_run_dir>
  3. Print a separator with the run dir, status, and best-epoch metrics.

At the end, prints a per-config status table.

Usage:
    uv run scripts/batch_train.py training/corner/configs/run_024.yaml \\
                                  training/corner/configs/run_025.yaml \\
                                  training/corner/configs/run_026.yaml
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "training" / "corner" / "runs"
TRAIN_SCRIPT = "scripts/train_corners.py"
VIZ_SCRIPT = "scripts/visualize_predictions.py"


def existing_run_dirs() -> set[Path]:
    if not RUNS_DIR.is_dir():
        return set()
    return {p for p in RUNS_DIR.iterdir()
            if p.is_dir() and p.name.startswith("run_")}


def detect_new_run(before: set[Path]) -> Path | None:
    """Return the run dir created since `before` snapshot, or None
    if none / multiple appeared. With sequential execution there
    should be exactly one new entry per train invocation."""
    after = existing_run_dirs()
    new = after - before
    if len(new) == 1:
        return next(iter(new))
    return None


def run_step(cmd: list[str], label: str) -> int:
    """Stream stdout/stderr through, return exit code."""
    print(f"\n>>> {label}: {' '.join(cmd)}")
    sys.stdout.flush()
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    return proc.returncode


def format_seconds(sec: float) -> str:
    if sec < 60:
        return f"{sec:.0f}s"
    m, s = divmod(int(sec), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def summarize_metrics(run_dir: Path) -> str:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        return "  (metrics.json missing -- training likely crashed)"
    m = json.loads(metrics_path.read_text())
    lines = [
        f"  best epoch:        {m.get('best_epoch', '?')}"
        f" / {m.get('epochs_run', '?')}",
        f"  best val loss:    {m.get('val_loss', float('nan')):.5f}",
        f"  train loss @best: {m.get('train_loss', float('nan')):.5f}",
        f"  mean corner err:  "
        f"{m.get('mean_corner_error_px', float('nan')):.2f} px "
        f"({m.get('mean_corner_error_pct', float('nan')) * 100:.2f}% diag)",
    ]
    per_pct = m.get("per_corner_error_pct", {})
    if per_pct:
        parts = [f"{name}={per_pct[name] * 100:.2f}%"
                 for name in ("TL", "TR", "BR", "BL") if name in per_pct]
        if parts:
            lines.append(f"  per-corner:       {'  '.join(parts)}")
    return "\n".join(lines)


def banner(text: str, char: str = "=", width: int = 72) -> str:
    return char * width + "\n" + text + "\n" + char * width


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("configs", nargs="+", type=Path,
                        help="Config YAMLs to train, in order.")
    parser.add_argument("--skip-viz", action="store_true",
                        help="Skip the visualize_predictions step after "
                             "each training (still records metrics).")
    args = parser.parse_args()

    missing = [c for c in args.configs if not c.is_file()]
    if missing:
        for c in missing:
            print(f"Config not found: {c}")
        sys.exit(1)

    print(banner(f"Batch: {len(args.configs)} config(s)"))
    for i, c in enumerate(args.configs, start=1):
        print(f"  {i}. {c}")
    print()

    results: list[dict] = []
    batch_start = time.time()

    for i, config in enumerate(args.configs, start=1):
        print(banner(f"[{i}/{len(args.configs)}] {config}"))
        cfg_start = time.time()
        before = existing_run_dirs()

        train_rc = run_step(
            ["uv", "run", TRAIN_SCRIPT, "--config", str(config)],
            label="train",
        )
        run_dir = detect_new_run(before)

        viz_rc: int | None = None
        if train_rc == 0 and run_dir is not None and not args.skip_viz:
            viz_rc = run_step(
                ["uv", "run", VIZ_SCRIPT, str(run_dir)],
                label="visualize",
            )

        elapsed = time.time() - cfg_start

        status = "OK"
        if train_rc != 0:
            status = f"FAILED (train rc={train_rc})"
        elif run_dir is None:
            status = "FAILED (no new run dir detected)"
        elif viz_rc not in (None, 0):
            status = f"OK (viz rc={viz_rc})"

        results.append({
            "config": config,
            "run_dir": run_dir,
            "status": status,
            "elapsed": elapsed,
        })

        print()
        print(banner(f"[{i}/{len(args.configs)}] DONE: {config.name}",
                     char="-"))
        print(f"  status:   {status}")
        print(f"  run dir:  {run_dir if run_dir else '(none)'}")
        print(f"  elapsed:  {format_seconds(elapsed)}")
        if run_dir is not None:
            print(summarize_metrics(run_dir))
        print()

    print(banner(f"Batch complete in {format_seconds(time.time() - batch_start)}"))
    ok_count = sum(1 for r in results if r["status"].startswith("OK"))
    print(f"  {ok_count}/{len(results)} runs OK")
    print()
    print(f"  {'config':<48} {'run_dir':<24} {'status'}")
    for r in results:
        rd = r["run_dir"].name if r["run_dir"] else "-"
        print(f"  {str(r['config'].name):<48} {rd:<24} {r['status']}")


if __name__ == "__main__":
    main()
