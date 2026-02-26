"""GeneT5-style logging for GRPO training.

Provides CSV + stdout loggers and a background memory watcher adapted
from GeneT5/lib/util/_memwatch.py for MLX Metal backend.
"""

from __future__ import annotations

import csv
import sys
import threading
import time
from pathlib import Path

import psutil


# ---------------------------------------------------------------------------
# GRPO training logger
# ---------------------------------------------------------------------------

GRPO_LOG_FIELDS = [
    "timestamp", "elapsed_sec", "epoch", "step", "total_steps",
    "reward_mean", "reward_std", "accuracy",
    "policy_loss", "kl_loss", "kl_mean", "total_loss",
    "lr", "gen_sec", "update_sec",
    "ram_used_gb", "metal_alloc_gb",
]


class GRPOLogger:
    """CSV + stdout logger for GRPO training metrics."""

    def __init__(self, output_dir: Path):
        self.csv_path = output_dir / "grpo_log.csv"
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=GRPO_LOG_FIELDS)
        self._writer.writeheader()
        self._file.flush()
        self._start = time.time()

    def log_step(self, **kwargs: object) -> None:
        row: dict[str, str] = {}
        for f in GRPO_LOG_FIELDS:
            val = kwargs.get(f)
            if val is None:
                row[f] = ""
            elif isinstance(val, float):
                row[f] = f"{val:.6f}"
            else:
                row[f] = str(val)

        if not row.get("timestamp"):
            row["timestamp"] = f"{time.time():.0f}"
        if not row.get("elapsed_sec"):
            row["elapsed_sec"] = f"{time.time() - self._start:.1f}"

        self._writer.writerow(row)
        self._file.flush()

        # Compact stdout line
        parts = [
            f"E{kwargs.get('epoch', '?')}",
            f"S{kwargs.get('step', '?')}/{kwargs.get('total_steps', '?')}",
            f"R={_fmt(kwargs.get('reward_mean'))}",
            f"acc={_fmt(kwargs.get('accuracy'))}",
            f"loss={_fmt(kwargs.get('total_loss'))}",
            f"kl={_fmt(kwargs.get('kl_mean'))}",
            f"lr={_fmt(kwargs.get('lr'), 8)}",
        ]
        print("  ".join(parts), file=sys.stderr)

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()


# ---------------------------------------------------------------------------
# Test-set assessment logger
# ---------------------------------------------------------------------------

ASSESS_FIELDS = [
    "timestamp", "epoch", "step",
    "accuracy", "precision", "recall", "f1", "mcc",
    "n_correct", "n_total",
]


class AssessmentLogger:
    """CSV logger for periodic test-set assessments."""

    def __init__(self, output_dir: Path):
        self.csv_path = output_dir / "assessment_log.csv"
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=ASSESS_FIELDS)
        self._writer.writeheader()
        self._file.flush()

    def log(self, **kwargs: object) -> None:
        row: dict[str, str] = {}
        for f in ASSESS_FIELDS:
            val = kwargs.get(f)
            if val is None:
                row[f] = ""
            elif isinstance(val, float):
                row[f] = f"{val:.4f}"
            else:
                row[f] = str(val)
        if not row.get("timestamp"):
            row["timestamp"] = f"{time.time():.0f}"
        self._writer.writerow(row)
        self._file.flush()

        print(
            f"  [ASSESS] E{kwargs.get('epoch')} S{kwargs.get('step')} "
            f"acc={_fmt(kwargs.get('accuracy'))} f1={_fmt(kwargs.get('f1'))} "
            f"mcc={_fmt(kwargs.get('mcc'))} ({kwargs.get('n_correct')}/{kwargs.get('n_total')})",
            file=sys.stderr,
        )

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()


# ---------------------------------------------------------------------------
# Memory watcher (adapted from GeneT5, MLX Metal backend)
# ---------------------------------------------------------------------------

MEM_FIELDS = [
    "timestamp", "elapsed_sec",
    "ram_used_gb", "ram_total_gb", "ram_pct", "ram_avail_gb",
    "swap_used_gb", "swap_total_gb", "swap_pct",
    "metal_alloc_gb", "metal_peak_gb", "metal_cache_gb",
]


class MemoryWatcher:
    """Background memory monitor with pressure-adaptive sampling.

    Adapted from GeneT5/lib/util/_memwatch.py for MLX Metal backend.
    Tracks RAM, swap, and MLX Metal memory to CSV.

    Sampling rates:
    - Normal: 30s
    - High (>80% RAM): 5s
    - Critical (>90% RAM): 2s
    """

    def __init__(
        self,
        log_path: Path,
        normal_interval: int = 30,
        high_interval: int = 5,
        critical_interval: int = 2,
    ):
        self.log_path = Path(log_path)
        self.normal_interval = normal_interval
        self.high_interval = high_interval
        self.critical_interval = critical_interval
        self._start: float | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._file = None
        self._writer = None

    def _get_stats(self) -> tuple[dict[str, str], float, float]:
        now = time.time()
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        stats: dict[str, str] = {
            "timestamp": f"{now:.0f}",
            "elapsed_sec": f"{now - self._start:.1f}",
            "ram_used_gb": f"{mem.used / 1e9:.2f}",
            "ram_total_gb": f"{mem.total / 1e9:.2f}",
            "ram_pct": f"{mem.percent:.1f}",
            "ram_avail_gb": f"{mem.available / 1e9:.2f}",
            "swap_used_gb": f"{swap.used / 1e9:.2f}",
            "swap_total_gb": f"{swap.total / 1e9:.2f}",
            "swap_pct": f"{swap.percent:.1f}",
        }

        try:
            import mlx.core as mx
            stats["metal_alloc_gb"] = f"{mx.get_active_memory() / 1e9:.2f}"
            stats["metal_peak_gb"] = f"{mx.get_peak_memory() / 1e9:.2f}"
            stats["metal_cache_gb"] = f"{mx.get_cache_memory() / 1e9:.2f}"
        except Exception:
            stats["metal_alloc_gb"] = ""
            stats["metal_peak_gb"] = ""
            stats["metal_cache_gb"] = ""

        return stats, mem.percent, swap.percent

    def _choose_interval(self, ram_pct: float, swap_pct: float) -> int:
        if ram_pct > 90:
            return self.critical_interval
        if ram_pct > 80 or swap_pct > 10:
            return self.high_interval
        return self.normal_interval

    def _loop(self) -> None:
        self._file = open(self.log_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=MEM_FIELDS)
        self._writer.writeheader()
        self._file.flush()

        while self._running:
            stats, ram_pct, swap_pct = self._get_stats()
            self._writer.writerow(stats)
            self._file.flush()
            interval = self._choose_interval(ram_pct, swap_pct)
            time.sleep(interval)

        self._file.close()
        self._file = None

    def start(self) -> None:
        if self._running:
            return
        self._start = time.time()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"  Memory watcher started: {self.log_path}", file=sys.stderr)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print(f"  Memory watcher stopped: {self.log_path}", file=sys.stderr)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __del__(self):
        if self._running:
            self._running = False
        if self._file and not self._file.closed:
            self._file.close()


def create_memory_watcher(output_dir: Path, prefix: str = "memory") -> MemoryWatcher:
    """Create a memory watcher logging to output_dir/prefix_TIMESTAMP.csv."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    return MemoryWatcher(Path(output_dir) / f"{prefix}_{ts}.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(val: object, precision: int = 4) -> str:
    if val is None:
        return "?"
    if isinstance(val, float):
        return f"{val:.{precision}f}"
    return str(val)
