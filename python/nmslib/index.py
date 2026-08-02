"""Dense-vector HNSW index with nmslib's public workflow and signatures."""

from __future__ import annotations

from enum import IntEnum
from concurrent.futures import ThreadPoolExecutor
import operator
from typing import Iterable

import numpy as np

from ._lib import addr, f64, lib


class DataType(IntEnum):
    DENSE_VECTOR = 0
    SPARSE_VECTOR = 1
    OBJECT_AS_STRING = 2


class DistType(IntEnum):
    FLOAT = 0
    INT = 1


_SPACES = {"l2": 0, "l1": 1, "cosinesimil": 2, "negdotprod": 3}
_BATCH_PARALLEL_WORK = 250_000
_BATCH_MIN_QUERIES = 16
_INT32_INFO = np.iinfo(np.int32)


def _id(value) -> int:
    """Validate IDs before they enter the int32 result ABI."""
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError("IDs must be integers") from exc
    if not _INT32_INFO.min <= result <= _INT32_INFO.max:
        raise ValueError("IDs must fit in signed int32")
    return result


def _normalize(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=-1, keepdims=True)
    return np.divide(rows, norms, out=np.zeros_like(rows), where=norms != 0.0)


class Index:
    """A single-process dense HNSW index.

    Graph construction follows the HNSW insertion procedure. The hot base-layer
    best-first traversal and all candidate distance calculations execute in
    Mojo; Python only owns graph construction and the public object contract.
    """

    def __init__(self, space: str = "cosinesimil", method: str = "hnsw", data_type=DataType.DENSE_VECTOR, dtype=DistType.FLOAT):
        if method != "hnsw":
            raise ValueError("only method='hnsw' is implemented")
        if space not in _SPACES:
            raise ValueError(f"unsupported dense-vector space {space!r}; supported: {', '.join(_SPACES)}")
        if DataType(data_type) != DataType.DENSE_VECTOR:
            raise ValueError("only DataType.DENSE_VECTOR is implemented")
        if DistType(dtype) != DistType.FLOAT:
            raise ValueError("only DistType.FLOAT is implemented")
        self.space = space
        self.method = method
        self.data_type = DataType(data_type)
        self.dtype = DistType(dtype)
        self._pending: list[np.ndarray] = []
        self._ids: list[int] = []
        self._data: np.ndarray | None = None
        self._levels: np.ndarray | None = None
        self._layers: list[list[list[int]]] | None = None
        self._base: np.ndarray | None = None
        self._entry = -1
        self._max_level = -1
        self._m = 16
        self._base_degree = 32
        self._ef_construction = 200
        self._ef_search = 100
        self._rng = np.random.default_rng(17)
        self._id_array: np.ndarray | None = None
        self._scratch: tuple[np.ndarray, ...] | None = None

    def addDataPoint(self, data, id: int | None = None):
        if self._data is not None:
            raise RuntimeError("addDataPoint must precede createIndex; create a new index to append")
        row = f64(data).reshape(-1)
        if not row.size:
            raise ValueError("data points must contain at least one value")
        if self._pending and row.size != self._pending[0].size:
            raise ValueError("all vectors must have the same dimension")
        point_id = _id(len(self._ids) if id is None else id)
        if point_id in self._ids:
            raise ValueError(f"duplicate id {point_id}")
        self._pending.append(row.copy())
        self._ids.append(point_id)

    def addDataPointBatch(self, data, ids: Iterable[int] | None = None, num_threads: int = 0):
        rows = f64(data)
        if rows.ndim != 2:
            raise ValueError("data must be a two-dimensional dense array")
        if not len(rows) or not rows.shape[1]:
            raise ValueError("data must contain at least one non-empty vector")
        if ids is None:
            ids = range(len(self._ids), len(self._ids) + len(rows))
        ids = [_id(value) for value in ids]
        if len(ids) != len(rows):
            raise ValueError("ids must have one value per row")
        if len(set(ids)) != len(ids) or any(point_id in self._ids for point_id in ids):
            raise ValueError("duplicate id")
        for row, point_id in zip(rows, ids):
            self.addDataPoint(row, int(point_id))

    def _distance_many(self, query: np.ndarray, nodes: Iterable[int]) -> list[tuple[float, int]]:
        nodes = list(nodes)
        if not nodes:
            return []
        block = self._data[nodes]
        if self.space == "l2":
            values = np.linalg.norm(block - query, axis=1)
        elif self.space == "l1":
            values = np.abs(block - query).sum(axis=1)
        elif self.space == "cosinesimil":
            values = 1.0 - block @ query
        else:
            values = -(block @ query)
        return [(float(value), node) for value, node in zip(values, nodes)]

    def _distance_nodes(self, a: int, b: int) -> float:
        return self._distance_many(self._data[a], [b])[0][0]

    def _greedy(self, query: np.ndarray, entry: int, layer: int) -> int:
        current = entry
        current_distance = self._distance_many(query, [current])[0][0]
        changed = True
        while changed:
            changed = False
            for neighbor in self._layers[layer][current]:
                value = self._distance_many(query, [neighbor])[0][0]
                if value < current_distance:
                    current, current_distance, changed = neighbor, value, True
        return current

    def _search_layer(self, query: np.ndarray, entries: list[int], ef: int, layer: int) -> list[int]:
        import heapq

        visited = set(entries)
        candidates = []
        best = []
        for value, node in self._distance_many(query, entries):
            heapq.heappush(candidates, (value, node))
            heapq.heappush(best, (-value, node))
        while candidates:
            value, node = heapq.heappop(candidates)
            if len(best) >= ef and value > -best[0][0]:
                break
            neighbors = self._layers[layer][node]
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                candidate_distance = self._distance_many(query, [neighbor])[0][0]
                if len(best) < ef or candidate_distance < -best[0][0]:
                    heapq.heappush(candidates, (candidate_distance, neighbor))
                    heapq.heappush(best, (-candidate_distance, neighbor))
                    if len(best) > ef:
                        heapq.heappop(best)
        return [node for _, node in sorted(((-value, node) for value, node in best))]

    def _connect(self, node: int, neighbors: list[int], layer: int):
        degree = self._base_degree if layer == 0 else self._m
        own = self._layers[layer][node]
        own.extend(neighbor for neighbor in neighbors if neighbor not in own)
        own[:] = [candidate for _, candidate in sorted((self._distance_nodes(node, candidate), candidate) for candidate in own)[:degree]]
        for neighbor in neighbors:
            links = self._layers[layer][neighbor]
            if node not in links:
                links.append(node)
            if len(links) > degree:
                links[:] = [candidate for _, candidate in sorted((self._distance_nodes(neighbor, candidate), candidate) for candidate in links)[:degree]]

    def createIndex(self, index_params: dict | None = None, print_progress: bool = False):
        if not self._pending:
            raise ValueError("cannot create an index with no vectors")
        params = index_params or {}
        self._m = max(2, int(params.get("M", 16)))
        self._base_degree = 2 * self._m
        self._ef_construction = max(self._m, int(params.get("efConstruction", 200)))
        self._rng = np.random.default_rng(int(params.get("seed", 17)))
        self._data = np.ascontiguousarray(np.vstack(self._pending), dtype=np.float64)
        if self.space == "cosinesimil":
            self._data = _normalize(self._data)
        n = len(self._data)
        self._levels = np.floor(-np.log(self._rng.random(n)) / np.log(self._m)).astype(np.int64)
        self._max_level = -1
        self._entry = -1
        self._layers = []
        for node, level in enumerate(self._levels):
            while len(self._layers) <= level:
                self._layers.append([[] for _ in range(n)])
            if self._entry < 0:
                self._entry = node
                self._max_level = int(level)
                continue
            entry = self._entry
            for layer in range(self._max_level, int(level), -1):
                entry = self._greedy(self._data[node], entry, layer)
            for layer in range(min(int(level), self._max_level), -1, -1):
                candidates = self._search_layer(self._data[node], [entry], self._ef_construction, layer)
                degree = self._base_degree if layer == 0 else self._m
                selected = candidates[:degree]
                self._connect(node, selected, layer)
                if candidates:
                    entry = candidates[0]
            if level > self._max_level:
                self._entry, self._max_level = node, int(level)
        self._base = np.full((n, self._base_degree), -1.0, dtype=np.float64)
        for node, links in enumerate(self._layers[0]):
            self._base[node, : len(links)] = links
        self._pending.clear()
        self._id_array = np.asarray(self._ids, dtype=np.int32)
        self._scratch = None
        return None

    def setQueryTimeParams(self, params: dict):
        if "efSearch" in params:
            self._ef_search = max(1, int(params["efSearch"]))

    def _prepared_query(self, query) -> np.ndarray:
        if self._data is None:
            raise RuntimeError("createIndex must be called before querying")
        vector = f64(query).reshape(-1)
        if not vector.size:
            raise ValueError("query must contain at least one value")
        if vector.size != self._data.shape[1]:
            raise ValueError(f"query dimension is {vector.size}, expected {self._data.shape[1]}")
        return _normalize(vector[None, :])[0] if self.space == "cosinesimil" else vector

    def _new_scratch(self, n: int) -> tuple[np.ndarray, ...]:
        return tuple(np.empty(n, dtype=np.float64) for _ in range(5))

    def _search_prepared(self, vector: np.ndarray, k: int, scratch: tuple[np.ndarray, ...]):
        n, d = self._data.shape
        entry = self._entry
        for layer in range(self._max_level, 0, -1):
            entry = self._greedy(vector, entry, layer)
        ef = min(n, max(k, self._ef_search))
        result_ids = np.empty(k, dtype=np.float64)
        result_distances = np.empty(k, dtype=np.float64)
        visited, candidate_ids, candidate_distances, top_ids, top_distances = scratch
        lib().mn_hnsw_search(addr(self._data), addr(self._base), addr(vector), addr(result_ids), addr(result_distances), addr(visited), addr(candidate_ids), addr(candidate_distances), addr(top_ids), addr(top_distances), n, d, self._base_degree, entry, ef, k, _SPACES[self.space])
        rows = result_ids.astype(np.int64)
        return self._id_array[rows], result_distances.astype(np.float32)

    def knnQuery(self, query, k: int = 10):
        vector = self._prepared_query(query)
        n = self._data.shape[0]
        k = min(max(1, operator.index(k)), n)
        if self._scratch is None or self._scratch[0].size != n:
            self._scratch = self._new_scratch(n)
        return self._search_prepared(vector, k, self._scratch)

    def knnQueryBatch(self, queries, k: int = 10, num_threads: int = 0):
        rows = f64(queries)
        if rows.ndim != 2:
            raise ValueError("queries must be a two-dimensional dense array")
        if self._data is None:
            raise RuntimeError("createIndex must be called before querying")
        n, d = self._data.shape
        if rows.shape[1] != d:
            raise ValueError(f"query dimension is {rows.shape[1]}, expected {d}")
        if not len(rows):
            return []
        k = min(max(1, operator.index(k)), n)
        prepared = _normalize(rows) if self.space == "cosinesimil" else rows
        workers = int(num_threads)
        work = len(prepared) * n * d
        if len(prepared) < _BATCH_MIN_QUERIES or work < _BATCH_PARALLEL_WORK or workers < 2:
            return [self.knnQuery(row, k) for row in prepared]

        workers = min(workers, len(prepared))
        chunk = (len(prepared) + workers - 1) // workers

        def search_chunk(start: int, stop: int):
            scratch = self._new_scratch(n)
            return [(row, self._search_prepared(prepared[row], k, scratch)) for row in range(start, stop)]

        spans = [(start, min(start + chunk, len(prepared))) for start in range(0, len(prepared), chunk)]
        with ThreadPoolExecutor(max_workers=len(spans)) as pool:
            chunks = list(pool.map(lambda span: search_chunk(*span), spans))
        results = [None] * len(prepared)
        for values in chunks:
            for row, value in values:
                results[row] = value
        return results

    def getDistance(self, id1: int, id2: int):
        if self._data is None:
            if not self._pending:
                raise RuntimeError("addDataPointBatch must be called before getDistance")
            data = np.ascontiguousarray(np.vstack(self._pending), dtype=np.float64)
            if self.space == "cosinesimil":
                data = _normalize(data)
            old_data, self._data = self._data, data
            try:
                return self.getDistance(id1, id2)
            finally:
                self._data = old_data
        lookup = {point_id: row for row, point_id in enumerate(self._ids)}
        try:
            return self._distance_nodes(lookup[int(id1)], lookup[int(id2)])
        except KeyError as exc:
            raise ValueError(f"unknown id {exc.args[0]}") from None

    def getDataPoint(self, id: int):
        if self._data is None:
            raise RuntimeError("createIndex must be called before getDataPoint")
        try:
            return self._data[self._ids.index(int(id))].copy()
        except ValueError:
            raise ValueError(f"unknown id {id}") from None

    def getCurrentCount(self):
        return len(self._ids)

    def getMaxElements(self):
        return len(self._ids)

    def freeIndex(self):
        self._layers = None
        self._base = None
        self._data = None
        self._levels = None
        self._pending.clear()
        self._ids.clear()
        self._id_array = None
        self._scratch = None
        self._entry = self._max_level = -1

    def saveIndex(self, filename: str, save_data: bool = False):
        if self._data is None:
            raise RuntimeError("createIndex must be called before saveIndex")
        offsets = [0]
        links: list[int] = []
        for layer in self._layers:
            for node_links in layer:
                links.extend(node_links)
                offsets.append(len(links))
        with open(filename, "wb") as handle:
            np.savez_compressed(handle, data=self._data, ids=np.asarray(self._ids), levels=self._levels, base=self._base, entry=self._entry, max_level=self._max_level, m=self._m, ef_construction=self._ef_construction, space=self.space, offsets=np.asarray(offsets, dtype=np.int64), links=np.asarray(links, dtype=np.int64))

    def loadIndex(self, filename: str, load_data: bool = False):
        with np.load(filename, allow_pickle=False) as saved:
            if str(saved["space"]) != self.space:
                raise ValueError("saved index space does not match this index")
            data = f64(saved["data"])
            ids = [_id(value) for value in saved["ids"]]
            levels = np.asarray(saved["levels"])
            base = np.asarray(saved["base"])
            entry, max_level = int(saved["entry"]), int(saved["max_level"])
            m, ef_construction = int(saved["m"]), int(saved["ef_construction"])
            offsets, links = np.asarray(saved["offsets"]), np.asarray(saved["links"])
        n = len(data)
        if data.ndim != 2 or not n or not data.shape[1] or len(ids) != n or len(set(ids)) != n:
            raise ValueError("invalid saved index dimensions or IDs")
        if levels.ndim != 1 or not np.issubdtype(levels.dtype, np.integer) or len(levels) != n or np.any(levels < 0):
            raise ValueError("invalid saved index levels")
        if m < 2 or ef_construction < m or max_level < 0 or entry < 0 or entry >= n:
            raise ValueError("invalid saved index parameters")
        if base.ndim != 2 or base.shape[0] != n or not base.shape[1]:
            raise ValueError("invalid saved base graph")
        if not np.issubdtype(base.dtype, np.number) or not np.isfinite(base).all() or np.any(base != np.trunc(base)) or np.any((base < -1) | (base >= n)):
            raise ValueError("invalid saved base graph")
        expected_offsets = (max_level + 1) * n + 1
        if offsets.ndim != 1 or len(offsets) != expected_offsets or not np.issubdtype(offsets.dtype, np.integer) or offsets[0] != 0 or offsets[-1] != len(links) or np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("invalid saved graph offsets")
        if links.ndim != 1 or not np.issubdtype(links.dtype, np.integer) or np.any((links < 0) | (links >= n)):
            raise ValueError("invalid saved graph links")
        self._data = data
        self._ids = ids
        self._id_array = np.asarray(ids, dtype=np.int32)
        self._levels = np.asarray(levels, dtype=np.int64)
        self._base = np.ascontiguousarray(base, dtype=np.float64)
        self._entry, self._max_level = entry, max_level
        self._m, self._base_degree = m, self._base.shape[1]
        self._ef_construction = ef_construction
        self._layers = [[[] for _ in range(n)] for _ in range(max_level + 1)]
        self._scratch = None
        cursor = 0
        for layer_number, layer in enumerate(self._layers):
            for node in range(len(self._data)):
                self._layers[layer_number][node] = [int(value) for value in links[offsets[cursor] : offsets[cursor + 1]]]
                cursor += 1
        return None


def init(space: str = "cosinesimil", space_params: dict | None = None, method: str = "hnsw", data_type=DataType.DENSE_VECTOR, dtype=DistType.FLOAT) -> Index:
    """Create an HNSW index; signature-compatible with ``nmslib.init``."""
    return Index(space=space, method=method, data_type=data_type, dtype=dtype)
