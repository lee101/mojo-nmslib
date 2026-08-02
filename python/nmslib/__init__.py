"""A dense-vector, HNSW-compatible subset of :mod:`nmslib` implemented in Mojo."""

from .index import DataType, DistType, Index, init

__all__ = ["DataType", "DistType", "Index", "init"]
__version__ = "0.1.0"
