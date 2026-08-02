"""Dense HNSW base-layer search kernels exposed through a small C ABI."""

from std.sys import simd_width_of

comptime W = simd_width_of[DType.float64]()
comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime INF = 1.7976931348623157e308


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


@always_inline
def distance(data: Ptr, query: Ptr, row: Int, d: Int, space: Int) -> Float64:
    """Distance code: 0=l2, 1=l1, 2=cosine on normalized vectors, 3=negative dot."""
    var base = data + row * d
    if space == 1:
        var acc = SIMD[DType.float64, W](0.0)
        var j = 0
        while j + W <= d:
            var delta = base.load[width=W](j) - query.load[width=W](j)
            acc += max(delta, -delta)
            j += W
        var total = acc.reduce_add()
        while j < d:
            var delta = base[j] - query[j]
            total += -delta if delta < 0.0 else delta
            j += 1
        return total
    if space == 3:
        var acc = SIMD[DType.float64, W](0.0)
        var j = 0
        while j + W <= d:
            acc += base.load[width=W](j) * query.load[width=W](j)
            j += W
        var total = acc.reduce_add()
        while j < d:
            total += base[j] * query[j]
            j += 1
        return -total
    if space == 2:
        var acc = SIMD[DType.float64, W](0.0)
        var j = 0
        while j + W <= d:
            acc += base.load[width=W](j) * query.load[width=W](j)
            j += W
        var total = acc.reduce_add()
        while j < d:
            total += base[j] * query[j]
            j += 1
        return 1.0 - total
    var acc = SIMD[DType.float64, W](0.0)
    var j = 0
    while j + W <= d:
        var delta = base.load[width=W](j) - query.load[width=W](j)
        acc += delta * delta
        j += W
    var total = acc.reduce_add()
    while j < d:
        var delta = base[j] - query[j]
        total += delta * delta
        j += 1
    # nmslib's HNSW ``l2`` query path reports squared L2, while getDistance
    # reports the Euclidean metric. Ranking is identical and the Python layer
    # preserves that API distinction.
    return total


@always_inline
def push_min(ids: Ptr, distances: Ptr, count: Int, node: Int, value: Float64):
    var slot = count
    ids[slot] = Float64(node)
    distances[slot] = value
    while slot > 0:
        var parent = (slot - 1) // 2
        if distances[parent] <= value:
            break
        ids[slot] = ids[parent]
        distances[slot] = distances[parent]
        slot = parent
    ids[slot] = Float64(node)
    distances[slot] = value


@always_inline
def push_max(ids: Ptr, distances: Ptr, count: Int, node: Int, value: Float64):
    var slot = count
    ids[slot] = Float64(node)
    distances[slot] = value
    while slot > 0:
        var parent = (slot - 1) // 2
        if distances[parent] >= value:
            break
        ids[slot] = ids[parent]
        distances[slot] = distances[parent]
        slot = parent
    ids[slot] = Float64(node)
    distances[slot] = value


@always_inline
def sift_down_min(ids: Ptr, distances: Ptr, count: Int):
    var slot = 0
    var node = ids[slot]
    var value = distances[slot]
    while True:
        var child = slot * 2 + 1
        if child >= count:
            break
        if child + 1 < count and distances[child + 1] < distances[child]:
            child += 1
        if distances[child] >= value:
            break
        ids[slot] = ids[child]
        distances[slot] = distances[child]
        slot = child
    ids[slot] = node
    distances[slot] = value


@always_inline
def sift_down_max(ids: Ptr, distances: Ptr, count: Int):
    var slot = 0
    var node = ids[slot]
    var value = distances[slot]
    while True:
        var child = slot * 2 + 1
        if child >= count:
            break
        if child + 1 < count and distances[child + 1] > distances[child]:
            child += 1
        if distances[child] <= value:
            break
        ids[slot] = ids[child]
        distances[slot] = distances[child]
        slot = child
    ids[slot] = node
    distances[slot] = value


@export("mn_hnsw_search")
def mn_hnsw_search(
    data_addr: Int,
    graph_addr: Int,
    query_addr: Int,
    result_ids_addr: Int,
    result_distances_addr: Int,
    visited_addr: Int,
    candidate_ids_addr: Int,
    candidate_distances_addr: Int,
    top_ids_addr: Int,
    top_distances_addr: Int,
    n: Int,
    d: Int,
    degree: Int,
    entry: Int,
    ef: Int,
    k: Int,
    space: Int,
) abi("C"):
    """Best-first search of one HNSW/NSW layer with caller-owned scratch.

    Candidate storage is a min-heap with n slots; the top set is a bounded
    max-heap with ef slots. Both are caller-owned and allocation-free.
    """
    if data_addr == 0 or graph_addr == 0 or query_addr == 0 or result_ids_addr == 0 or result_distances_addr == 0 or visited_addr == 0 or candidate_ids_addr == 0 or candidate_distances_addr == 0 or top_ids_addr == 0 or top_distances_addr == 0:
        return
    if n <= 0 or d <= 0 or degree <= 0 or ef <= 0 or k <= 0 or k > n or space < 0 or space > 3:
        return
    var data = p(data_addr)
    var graph = p(graph_addr)
    var query = p(query_addr)
    var result_ids = p(result_ids_addr)
    var result_distances = p(result_distances_addr)
    var visited = p(visited_addr)
    var candidate_ids = p(candidate_ids_addr)
    var candidate_distances = p(candidate_distances_addr)
    var top_ids = p(top_ids_addr)
    var top_distances = p(top_distances_addr)
    for i in range(n):
        visited[i] = 0.0
    var start = entry
    if start < 0 or start >= n:
        start = 0
    var initial = distance(data, query, start, d, space)
    visited[start] = 1.0
    var candidate_count = 1
    var top_count = 1
    candidate_ids[0] = Float64(start)
    candidate_distances[0] = initial
    top_ids[0] = Float64(start)
    top_distances[0] = initial
    while True:
        if candidate_count == 0:
            break
        var best_distance = candidate_distances[0]
        var current = Int(candidate_ids[0])
        candidate_count -= 1
        if candidate_count > 0:
            candidate_ids[0] = candidate_ids[candidate_count]
            candidate_distances[0] = candidate_distances[candidate_count]
            sift_down_min(candidate_ids, candidate_distances, candidate_count)
        if top_count >= ef and best_distance > top_distances[0]:
            break
        for slot in range(degree):
            var neighbor = Int(graph[current * degree + slot])
            if neighbor < 0 or neighbor >= n or visited[neighbor] != 0.0:
                continue
            visited[neighbor] = 1.0
            var value = distance(data, query, neighbor, d, space)
            if top_count < ef or value < top_distances[0]:
                if candidate_count < n:
                    push_min(candidate_ids, candidate_distances, candidate_count, neighbor, value)
                    candidate_count += 1
                if top_count < ef:
                    push_max(top_ids, top_distances, top_count, neighbor, value)
                    top_count += 1
                else:
                    top_ids[0] = Float64(neighbor)
                    top_distances[0] = value
                    sift_down_max(top_ids, top_distances, top_count)
    for i in range(1, top_count):
        var node = top_ids[i]
        var value = top_distances[i]
        var slot = i
        while slot > 0 and top_distances[slot - 1] > value:
            top_ids[slot] = top_ids[slot - 1]
            top_distances[slot] = top_distances[slot - 1]
            slot -= 1
        top_ids[slot] = node
        top_distances[slot] = value
    var answer_count = k
    if answer_count > top_count:
        answer_count = top_count
    for i in range(answer_count):
        result_ids[i] = top_ids[i]
        result_distances[i] = top_distances[i]
    for i in range(answer_count, k):
        result_ids[i] = -1.0
        result_distances[i] = INF


@export("mn_distances")
def mn_distances(
    data_addr: Int, query_addr: Int, output_addr: Int, n: Int, d: Int, space: Int
) abi("C"):
    if data_addr == 0 or query_addr == 0 or output_addr == 0 or n <= 0 or d <= 0 or space < 0 or space > 3:
        return
    var data = p(data_addr)
    var query = p(query_addr)
    var output = p(output_addr)
    for row in range(n):
        output[row] = distance(data, query, row, d, space)
