"""HNSW query benchmark; run only through ``pixi run bench`` for its flock."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import nmslib


def best_time(fn, repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        started = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - started)
    return best


def upstream_time(root: Path, space: str, x: np.ndarray, q: np.ndarray) -> float:
    payload = root / "bench" / ".reference-data.npz"
    np.savez(payload, x=x.astype(np.float32), q=q.astype(np.float32))
    code = r'''
import json, sys, time
import numpy as np
import nmslib
p = np.load(sys.argv[1])
i = nmslib.init(space=sys.argv[2], method="hnsw")
i.addDataPointBatch(p["x"])
i.createIndex({"M": 16, "efConstruction": 120})
i.setQueryTimeParams({"efSearch": 100})
i.knnQueryBatch(p["q"][:2], k=10, num_threads=1)
best = float("inf")
for _ in range(3):
    start = time.perf_counter()
    i.knnQueryBatch(p["q"], k=10, num_threads=1)
    best = min(best, time.perf_counter() - start)
print(json.dumps(best))
'''
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    output = subprocess.check_output([sys.executable, "-c", code, str(payload), space], cwd="/tmp", env=env, text=True)
    payload.unlink(missing_ok=True)
    return float(json.loads(output))


def run_case(space: str, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(400, 32))
    q = rng.normal(size=(100, 32))
    ours = nmslib.init(space=space)
    ours.addDataPointBatch(x)
    ours.createIndex({"M": 16, "efConstruction": 120, "seed": 17})
    ours.setQueryTimeParams({"efSearch": 100})
    ours.knnQueryBatch(q[:2], k=10)
    mojo_seconds = best_time(lambda: ours.knnQueryBatch(q, k=10))
    upstream_seconds = upstream_time(Path(__file__).resolve().parents[1], space, x, q)
    ratio = upstream_seconds / mojo_seconds
    verdict = "faster" if ratio > 1 else "slower"
    print(f"| HNSW `{space}` query (400 x 32, 100 queries) | {mojo_seconds * 1e3:.1f} ms | {upstream_seconds * 1e3:.1f} ms | {ratio:.2f}x {verdict} |")


def main():
    print("| case | mojo-nmslib | nmslib | result |")
    print("| --- | ---: | ---: | --- |")
    run_case("l2", 0)
    run_case("negdotprod", 1)


if __name__ == "__main__":
    main()
