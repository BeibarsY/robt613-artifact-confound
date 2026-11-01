"""Wall-time logging to results/timing.csv."""
from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime

import config


@contextmanager
def timed(stage: str):
    """Time a block and append ``timestamp,mode,stage,seconds`` to results/timing.csv."""
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    path = config.RESULTS / "timing.csv"
    new = not path.exists()
    with path.open("a") as f:
        if new:
            f.write("timestamp,mode,stage,seconds\n")
        f.write(f"{datetime.now().isoformat(timespec='seconds')},{config.MODE},{stage},{dt:.1f}\n")
    print(f"[timing] {stage}: {dt:.1f} s")
