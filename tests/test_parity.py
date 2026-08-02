"""Parity checks against the real nmslib package plus exact dense references."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import nmslib
from nmslib._lib import addr, lib


UPSTREAM = r'''
import json, sys
import numpy as np
import nmslib
p = json.load(sys.stdin)
x = np.asarray(p["x"], dtype=np.float32)
q = np.asarray(p["q"], dtype=np.float32)
i = nmslib.init(space=p["space"], method="hnsw")
i.addDataPointBatch(x, ids=p["ids"])
d = [float(i.getDistance(p["ids"][0], value)) for value in p["ids"]]
i.createIndex({"M": 12, "efConstruction": 100})
i.setQueryTimeParams({"efSearch": 100})
ids, distances = i.knnQuery(q, p["k"])
print(json.dumps({"distance": d, "ids": ids.tolist(), "query": distances.tolist()}))
'''


def upstream(tmp_path: Path, space: str, x: np.ndarray, q: np.ndarray, ids, k: int):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    output = subprocess.check_output(
        [sys.executable, "-c", UPSTREAM],
        input=json.dumps({"space": space, "x": x.tolist(), "q": q.tolist(), "ids": list(ids), "k": k}),
        text=True,
        cwd=tmp_path,
        env=environment,
    )
    return json.loads(output)


@pytest.mark.parametrize("space", ["l2", "l1", "cosinesimil", "negdotprod"])
def test_distances_and_query_match_upstream_on_published_spaces(tmp_path, space):
    x = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0], [-2.0, 1.0]])
    ids = [0, 1, 2, 3]
    q = np.array([1.0, 0.0])
    theirs = upstream(tmp_path, space, x, q, ids, 3)
    ours = nmslib.init(space=space)
    ours.addDataPointBatch(x, ids=ids)
    assert np.allclose([ours.getDistance(ids[0], value) for value in ids], theirs["distance"], atol=2e-6)
    ours.createIndex({"M": 12, "efConstruction": 100})
    ours.setQueryTimeParams({"efSearch": 100})
    found_ids, found_distances = ours.knnQuery(q, 3)
    assert np.array_equal(found_ids, np.asarray(theirs["ids"], dtype=np.int32))
    assert np.allclose(found_distances, theirs["query"], atol=2e-6)


@pytest.mark.parametrize("space", ["l2", "l1", "cosinesimil", "negdotprod"])
def test_hnsw_recall_and_returned_distances(space):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(400, 16))
    queries = rng.normal(size=(6, 16))
    index = nmslib.init(space=space)
    index.addDataPointBatch(x)
    index.createIndex({"M": 16, "efConstruction": 120, "seed": 5})
    index.setQueryTimeParams({"efSearch": 160})
    recalls = []
    for query in queries:
        found, values = index.knnQuery(query, 10)
        if space == "l2":
            exact_values = ((x - query) ** 2).sum(axis=1)
        elif space == "l1":
            exact_values = np.abs(x - query).sum(axis=1)
        elif space == "cosinesimil":
            x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
            q_norm = query / np.linalg.norm(query)
            exact_values = 1.0 - x_norm @ q_norm
        else:
            exact_values = -(x @ query)
        truth = np.argsort(exact_values)[:10]
        recalls.append(len(set(found).intersection(truth)) / 10)
        assert np.allclose(values, exact_values[found], atol=2e-5)
    assert np.mean(recalls) >= 0.97


def test_batch_custom_ids_and_query_parameters():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(200, 8))
    ids = np.arange(1000, 1200) * 3
    index = nmslib.init(space="l2")
    index.addDataPointBatch(x, ids=ids, num_threads=2)
    index.createIndex({"M": 10, "efConstruction": 80})
    index.setQueryTimeParams({"efSearch": 80})
    batch = index.knnQueryBatch(x[:5], k=4, num_threads=2)
    assert len(batch) == 5
    assert all(result[0][0] == ids[row] and result[1][0] == 0.0 for row, result in enumerate(batch))
    assert index.getCurrentCount() == 200
    assert index.getMaxElements() == 200


@pytest.mark.parametrize("space, expected", [
    (0, lambda data, query: ((data - query) ** 2).sum(axis=1)),
    (1, lambda data, query: np.abs(data - query).sum(axis=1)),
    (2, lambda data, query: 1.0 - data @ query),
    (3, lambda data, query: -(data @ query)),
])
def test_simd_distance_tail_matches_numpy(space, expected):
    rng = np.random.default_rng(123)
    data = rng.normal(size=(9, 7)).astype(np.float64)
    query = rng.normal(size=7).astype(np.float64)
    output = np.empty(len(data), dtype=np.float64)
    lib().mn_distances(addr(data), addr(query), addr(output), len(data), data.shape[1], space)
    assert np.allclose(output, expected(data, query))


def test_large_explicit_parallel_batch_matches_serial():
    rng = np.random.default_rng(71)
    data = rng.normal(size=(32, 512))
    queries = rng.normal(size=(16, 512))
    index = nmslib.init(space="negdotprod")
    index.addDataPointBatch(data)
    index.createIndex({"M": 8, "efConstruction": 40, "seed": 3})
    index.setQueryTimeParams({"efSearch": 32})
    serial = index.knnQueryBatch(queries, k=5, num_threads=1)
    parallel = index.knnQueryBatch(queries, k=5, num_threads=2)
    for left, right in zip(serial, parallel):
        assert np.array_equal(left[0], right[0])
        assert np.allclose(left[1], right[1])


def test_save_load_round_trip(tmp_path):
    rng = np.random.default_rng(3)
    x, q = rng.normal(size=(300, 12)), rng.normal(size=12)
    first = nmslib.init(space="negdotprod")
    first.addDataPointBatch(x)
    first.createIndex({"M": 12, "efConstruction": 100})
    first.setQueryTimeParams({"efSearch": 150})
    expected = first.knnQuery(q, 8)
    filename = tmp_path / "index.nms"
    first.saveIndex(filename, save_data=True)
    second = nmslib.init(space="negdotprod")
    second.loadIndex(filename, load_data=True)
    second.setQueryTimeParams({"efSearch": 150})
    received = second.knnQuery(q, 8)
    assert np.array_equal(received[0], expected[0])
    assert np.allclose(received[1], expected[1])


def test_load_rejects_malformed_graph_before_ffi(tmp_path):
    filename = tmp_path / "bad.nms"
    with open(filename, "wb") as handle:
        np.savez(handle, data=np.ones((1, 2)), ids=np.array([0]), levels=np.array([0]),
                 base=np.array([[2.0]]), entry=0, max_level=0, m=2,
                 ef_construction=2, space="l2", offsets=np.array([0, 0]), links=np.array([]))
    with pytest.raises(ValueError, match="base graph"):
        nmslib.init(space="l2").loadIndex(filename)


def test_contract_errors_and_enums():
    assert nmslib.DataType.DENSE_VECTOR.value == 0
    assert nmslib.DistType.FLOAT.value == 0
    with pytest.raises(ValueError, match="method"):
        nmslib.init(method="sw-graph")
    with pytest.raises(ValueError, match="space"):
        nmslib.init(space="leven")
    index = nmslib.init(space="l2")
    with pytest.raises(ValueError, match="no vectors"):
        index.createIndex()
    index.addDataPoint([1, 2])
    with pytest.raises(ValueError, match="dimension"):
        index.addDataPoint([1, 2, 3])


def test_public_data_and_lifecycle_methods_and_lossy_inputs():
    index = nmslib.init(space="l2")
    index.addDataPoint([0, 0], id=9)
    index.addDataPoint([3, 4], id=11)
    assert index.getCurrentCount() == index.getMaxElements() == 2
    assert index.getDistance(9, 11) == pytest.approx(5.0)
    index.createIndex({"M": 2, "efConstruction": 4})
    assert np.array_equal(index.getDataPoint(11), np.array([3.0, 4.0]))
    index.freeIndex()
    assert index.getCurrentCount() == index.getMaxElements() == 0
    index.addDataPoint([1, 1])
    index.createIndex()
    assert index.knnQuery([1, 1], 1)[0][0] == 0
    with pytest.raises(TypeError):
        nmslib.init().addDataPoint(np.array([1 + 2j]))
    with pytest.raises(ValueError, match="int32"):
        nmslib.init().addDataPoint([1.0], id=2**31)
