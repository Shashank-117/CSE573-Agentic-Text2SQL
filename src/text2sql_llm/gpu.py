"""Peak GPU memory sampling via nvidia-smi (works for out-of-process servers).

This measures *device-level* memory in use, which includes other processes
(e.g. the desktop). Reports therefore also record the idle baseline so the
model's share can be estimated. Returns None when no GPU is visible —
callers must report that as "not measured", never as 0.
"""

from __future__ import annotations

import subprocess
import threading


def _query_used_mb() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        values = [float(v) for v in out.stdout.split() if v.strip()]
        return max(values) if values else None
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return None


def gpu_memory_used_mb() -> float | None:
    """Current device memory in use (MiB), or None if unavailable."""
    return _query_used_mb()


class GpuPeakSampler:
    """Context manager that polls device memory and keeps the peak."""

    def __init__(self, interval_s: float = 0.25, enabled: bool = True):
        self.interval_s = interval_s
        self.enabled = enabled and _query_used_mb() is not None
        self.peak_mb: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        value = _query_used_mb()
        if value is not None:
            self.peak_mb = value if self.peak_mb is None else max(self.peak_mb, value)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def __enter__(self) -> "GpuPeakSampler":
        if self.enabled:
            self._sample()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2)
            self._sample()
