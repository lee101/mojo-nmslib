# mojo-nmslib

`mojo-nmslib` is a standalone Mojo implementation of the dense-vector HNSW
workflow from [nmslib](https://github.com/nmslib/nmslib), with a Python module
named `nmslib` for migration of the covered subset. It is aimed at approximate
nearest-neighbour search in non-metric spaces, especially maximum inner-product
search through `negdotprod`.

The real upstream `nmslib` package is available on conda-forge and is included
in this repository's Pixi environment. Tests compare this port directly to its
distance and query results on fixed test vectors, then validate HNSW recall and
returned distances against exact NumPy search.

## Covered API

| API | Status |
| --- | --- |
| `init(space, space_params, method, data_type, dtype)` | `method="hnsw"`, dense float vectors; `space_params` is accepted but has no options in this port |
| `addDataPoint`, `addDataPointBatch` | implemented, including custom IDs |
| `createIndex(index_params, print_progress)` | HNSW with `M`, `efConstruction`, and `seed` |
| `setQueryTimeParams` | `efSearch` |
| `knnQuery`, `knnQueryBatch` | implemented; IDs are `int32`, distances are `float32` as upstream returns |
| `getDistance`, `getDataPoint`, `saveIndex`, `loadIndex`, `freeIndex` | implemented |
| spaces | `l2`, `l1`, `cosinesimil`, `negdotprod` |

`negdotprod` is deliberately included: it is non-metric maximum-inner-product
search, where a query's nearest result has the greatest dot product. As in
upstream nmslib, HNSW `l2` query distances are squared L2 while `getDistance`
returns the Euclidean distance.

Sparse vectors, string/object spaces (Levenshtein, Jaccard, etc.), integer
distances, other index methods, deletions, and upstream's on-disk binary format
are not covered. `saveIndex` and `loadIndex` round-trip this port's own compact
NumPy archive; they do not read upstream nmslib index files. The `save_data` and
`load_data` arguments are accepted for call compatibility; this archive always
includes the data needed to query it.

## Install and use

```bash
pixi install
pixi run build
```

Pixi activates `python/`, so this runs directly:

```python
import numpy as np
import nmslib

vectors = np.array([[1., 0.], [0., 1.], [2., 0.], [-1., 0.]])
index = nmslib.init(space="negdotprod", method="hnsw")
index.addDataPointBatch(vectors)
index.createIndex({"M": 12, "efConstruction": 100})
index.setQueryTimeParams({"efSearch": 100})

ids, distances = index.knnQuery(np.array([1., 0.]), k=2)
print(ids)        # [2 0]
print(distances)  # [-2. -1.]
```

Run the verification and benchmark suite with:

```bash
pixi run test
pixi run bench
```

## Benchmark

Measured by `pixi run bench` on Linux 6.8.0-136-generic, dual Intel Xeon
E5-2697 v4 (72 logical CPUs), Python 3.13.14, Mojo
1.1.0.dev2026081105, and nmslib 2.1.1.
The benchmark isolates query time after building each index, uses one upstream
query thread, and keeps the task's machine-wide `flock`.

| case | mojo-nmslib | nmslib | result |
| --- | ---: | ---: | --- |
| HNSW `l2` query (400 x 32, 100 queries) | 5.5 ms | 3.2 ms | 0.58x slower |
| HNSW `negdotprod` query (400 x 32, 100 queries) | 5.9 ms | 3.2 ms | 0.53x slower |

This port is still slower than mature upstream nmslib for complete index queries,
but query batches now cross the ABI once and execute all HNSW layers in Mojo.
The table reports that remaining gap directly.

## How it works

`src/capi.mojo` is one compilation unit containing four-way unrolled,
architecture-width SIMD dense-distance kernels and allocation-free traversal of
every HNSW layer. The candidate frontier is a min-heap and the bounded result set
is a max-heap; only the requested top `k` is ordered when `k` is much smaller than
`efSearch`. Query batches cross ctypes once, reuse typed scratch buffers, and write
`int32` IDs and `float32` distances directly into their final NumPy arrays. Batches
parallelize only at 16 queries and 250,000 scalar distance elements; smaller work
stays serial to avoid launch overhead.

There is intentionally no GPU path. Dense distance moves two float64 values for
at most three arithmetic operations per element, and graph traversal is
pointer-irregular, so both are well below the arithmetic intensity at which GPU
transfer and launch overhead can win.

The ABI uses `@export` functions with `abi("C")`. Every contiguous NumPy buffer
crosses ctypes as its integer address and is rebuilt as a typed Mojo pointer; no
data is copied at the FFI boundary and Mojo never owns or frees Python memory.

## License

MIT
