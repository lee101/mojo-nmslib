"""ctypes bridge for the one Mojo compilation unit."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-nmslib.so")
I = ctypes.c_int64

_SIGNATURES = {
    "mn_hnsw_search_batch": ([I] * 22, None),
    "mn_distances": ([I] * 6, None),
}


def build() -> str:
    sources = [os.path.join(ROOT, "src", name) for name in os.listdir(os.path.join(ROOT, "src")) if name.endswith(".mojo")]
    if os.path.exists(LIB) and os.path.getmtime(LIB) >= max(map(os.path.getmtime, sources)):
        return LIB
    command = shutil.which("mojo")
    if not command:
        raise RuntimeError("mojo is not on PATH; run through pixi or build with pixi run build")
    proc = subprocess.run([command, "build", "--emit", "shared-lib", os.path.join(ROOT, "src", "capi.mojo"), "-o", LIB], capture_output=True, text=True, timeout=1800)
    if proc.returncode:
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB


_loaded: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_loaded, name)
            fn.argtypes, fn.restype = argtypes, restype
    return _loaded


def f64(array) -> np.ndarray:
    """Return a finite, C-contiguous float64 array without lossy coercion."""
    value = np.asarray(array)
    if value.dtype.kind == "b" or value.dtype.kind in "cO" or value.dtype.kind not in "fiu":
        raise TypeError("dense vectors must contain real numeric values")
    if value.dtype.kind in "iu":
        # float64 exactly represents integers only through this limit.
        if value.size and np.max(np.abs(value.astype(object))) > 2**53:
            raise ValueError("integer vector values must be exactly representable as float64")
    elif value.dtype.itemsize > np.dtype(np.float64).itemsize:
        raise TypeError("dense vectors must use float32 or float64 values")
    value = np.ascontiguousarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("dense vectors must contain only finite values")
    return value


def addr(array: np.ndarray) -> int:
    if not array.flags.c_contiguous or array.size == 0:
        raise ValueError("FFI buffers must be non-empty C-contiguous arrays")
    return int(array.ctypes.data)
