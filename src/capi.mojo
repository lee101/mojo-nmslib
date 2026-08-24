"""Dense HNSW base-layer search kernels exposed through a small C ABI."""

from std.sys import simd_width_of
from max.algorithm import parallelize

comptime W = simd_width_of[DType.float64]()
comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime F32Ptr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime I32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime INF = 1.7976931348623157e308
comptime PARALLEL_WORK = 250_000
comptime MIN_PARALLEL_QUERIES = 16
comptime MAX_WORKERS = 16


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def i32p(addr: Int) -> I32Ptr:
    return I32Ptr(unsafe_from_address=addr)


def f32p(addr: Int) -> F32Ptr:
    return F32Ptr(unsafe_from_address=addr)


def u8p(addr: Int) -> U8Ptr:
    return U8Ptr(unsafe_from_address=addr)


@always_inline
def distance(data: Ptr, query: Ptr, row: Int, d: Int, space: Int) -> Float64:
    """Distance code: 0=l2, 1=l1, 2=cosine on normalized vectors, 3=negative dot."""
    var base = data + row * d
    if space == 1:
        var acc0 = SIMD[DType.float64, W](0.0)
        var acc1 = SIMD[DType.float64, W](0.0)
        var acc2 = SIMD[DType.float64, W](0.0)
        var acc3 = SIMD[DType.float64, W](0.0)
        var j = 0
        while j + 4 * W <= d:
            var delta0 = base.load[width=W](j) - query.load[width=W](j)
            var delta1 = base.load[width=W](j + W) - query.load[width=W](j + W)
            var delta2 = base.load[width=W](j + 2 * W) - query.load[width=W](j + 2 * W)
            var delta3 = base.load[width=W](j + 3 * W) - query.load[width=W](j + 3 * W)
            acc0 += max(delta0, -delta0)
            acc1 += max(delta1, -delta1)
            acc2 += max(delta2, -delta2)
            acc3 += max(delta3, -delta3)
            j += 4 * W
        var total = (acc0 + acc1 + acc2 + acc3).reduce_add()
        while j + W <= d:
            var delta = base.load[width=W](j) - query.load[width=W](j)
            total += max(delta, -delta).reduce_add()
            j += W
        while j < d:
            var delta = base[j] - query[j]
            total += -delta if delta < 0.0 else delta
            j += 1
        return total
    if space == 3:
        var acc0 = SIMD[DType.float64, W](0.0)
        var acc1 = SIMD[DType.float64, W](0.0)
        var acc2 = SIMD[DType.float64, W](0.0)
        var acc3 = SIMD[DType.float64, W](0.0)
        var j = 0
        while j + 4 * W <= d:
            acc0 += base.load[width=W](j) * query.load[width=W](j)
            acc1 += base.load[width=W](j + W) * query.load[width=W](j + W)
            acc2 += base.load[width=W](j + 2 * W) * query.load[width=W](j + 2 * W)
            acc3 += base.load[width=W](j + 3 * W) * query.load[width=W](j + 3 * W)
            j += 4 * W
        var total = (acc0 + acc1 + acc2 + acc3).reduce_add()
        while j + W <= d:
            total += (base.load[width=W](j) * query.load[width=W](j)).reduce_add()
            j += W
        while j < d:
            total += base[j] * query[j]
            j += 1
        return -total
    if space == 2:
        var acc0 = SIMD[DType.float64, W](0.0)
        var acc1 = SIMD[DType.float64, W](0.0)
        var acc2 = SIMD[DType.float64, W](0.0)
        var acc3 = SIMD[DType.float64, W](0.0)
        var j = 0
        while j + 4 * W <= d:
            acc0 += base.load[width=W](j) * query.load[width=W](j)
            acc1 += base.load[width=W](j + W) * query.load[width=W](j + W)
            acc2 += base.load[width=W](j + 2 * W) * query.load[width=W](j + 2 * W)
            acc3 += base.load[width=W](j + 3 * W) * query.load[width=W](j + 3 * W)
            j += 4 * W
        var total = (acc0 + acc1 + acc2 + acc3).reduce_add()
        while j + W <= d:
            total += (base.load[width=W](j) * query.load[width=W](j)).reduce_add()
            j += W
        while j < d:
            total += base[j] * query[j]
            j += 1
        return 1.0 - total
    var acc0 = SIMD[DType.float64, W](0.0)
    var acc1 = SIMD[DType.float64, W](0.0)
    var acc2 = SIMD[DType.float64, W](0.0)
    var acc3 = SIMD[DType.float64, W](0.0)
    var j = 0
    while j + 4 * W <= d:
        var delta0 = base.load[width=W](j) - query.load[width=W](j)
        var delta1 = base.load[width=W](j + W) - query.load[width=W](j + W)
        var delta2 = base.load[width=W](j + 2 * W) - query.load[width=W](j + 2 * W)
        var delta3 = base.load[width=W](j + 3 * W) - query.load[width=W](j + 3 * W)
        acc0 += delta0 * delta0
        acc1 += delta1 * delta1
        acc2 += delta2 * delta2
        acc3 += delta3 * delta3
        j += 4 * W
    var total = (acc0 + acc1 + acc2 + acc3).reduce_add()
    while j + W <= d:
        var delta = base.load[width=W](j) - query.load[width=W](j)
        total += (delta * delta).reduce_add()
        j += W
    while j < d:
        var delta = base[j] - query[j]
        total += delta * delta
        j += 1
    # nmslib's HNSW ``l2`` query path reports squared L2, while getDistance
    # reports the Euclidean metric. Ranking is identical and the Python layer
    # preserves that API distinction.
    return total


@always_inline
def push_min(ids: I32Ptr, distances: Ptr, count: Int, node: Int, value: Float64):
    var slot = count
    ids[slot] = Int32(node)
    distances[slot] = value
    while slot > 0:
        var parent = (slot - 1) // 2
        if distances[parent] <= value:
            break
        ids[slot] = ids[parent]
        distances[slot] = distances[parent]
        slot = parent
    ids[slot] = Int32(node)
    distances[slot] = value


@always_inline
def push_max(ids: I32Ptr, distances: Ptr, count: Int, node: Int, value: Float64):
    var slot = count
    ids[slot] = Int32(node)
    distances[slot] = value
    while slot > 0:
        var parent = (slot - 1) // 2
        if distances[parent] >= value:
            break
        ids[slot] = ids[parent]
        distances[slot] = distances[parent]
        slot = parent
    ids[slot] = Int32(node)
    distances[slot] = value


@always_inline
def sift_down_min(ids: I32Ptr, distances: Ptr, count: Int):
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
def sift_down_max(ids: I32Ptr, distances: Ptr, count: Int):
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


def search_one(
    data: Ptr,
    base_graph: I32Ptr,
    upper_graph: I32Ptr,
    queries: Ptr,
    result_ids: I32Ptr,
    result_distances: F32Ptr,
    visited_storage: U8Ptr,
    candidate_ids_storage: I32Ptr,
    candidate_distances_storage: Ptr,
    top_ids_storage: I32Ptr,
    top_distances_storage: Ptr,
    n: Int,
    d: Int,
    degree: Int,
    upper_degree: Int,
    max_level: Int,
    entry: Int,
    ef: Int,
    k: Int,
    space: Int,
    query_row: Int,
    scratch_row: Int,
):
    var query = queries + query_row * d
    var start = entry
    if start < 0 or start >= n:
        start = 0
    for layer_offset in range(max_level - 1, -1, -1):
        var current_distance = distance(data, query, start, d, space)
        var changed = True
        while changed:
            changed = False
            var graph_offset = layer_offset * n * upper_degree + start * upper_degree
            for slot in range(upper_degree):
                var neighbor = Int(upper_graph[graph_offset + slot])
                if neighbor < 0 or neighbor >= n:
                    continue
                var value = distance(data, query, neighbor, d, space)
                if value < current_distance:
                    start = neighbor
                    current_distance = value
                    changed = True

    var scratch_offset = scratch_row * n
    var visited = visited_storage + scratch_offset
    var candidate_ids = candidate_ids_storage + scratch_offset
    var candidate_distances = candidate_distances_storage + scratch_offset
    var top_ids = top_ids_storage + scratch_offset
    var top_distances = top_distances_storage + scratch_offset
    for i in range(n):
        visited[i] = 0
    var initial = distance(data, query, start, d, space)
    visited[start] = 1
    var candidate_count = 1
    var top_count = 1
    candidate_ids[0] = Int32(start)
    candidate_distances[0] = initial
    top_ids[0] = Int32(start)
    top_distances[0] = initial
    while candidate_count > 0:
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
            var neighbor = Int(base_graph[current * degree + slot])
            if neighbor < 0 or neighbor >= n or visited[neighbor] != 0:
                continue
            visited[neighbor] = 1
            var value = distance(data, query, neighbor, d, space)
            if top_count < ef or value < top_distances[0]:
                if candidate_count < n:
                    push_min(candidate_ids, candidate_distances, candidate_count, neighbor, value)
                    candidate_count += 1
                if top_count < ef:
                    push_max(top_ids, top_distances, top_count, neighbor, value)
                    top_count += 1
                else:
                    top_ids[0] = Int32(neighbor)
                    top_distances[0] = value
                    sift_down_max(top_ids, top_distances, top_count)
    var answer_count = min(k, top_count)
    var result_offset = query_row * k
    if k * 2 < top_count:
        for i in range(k):
            candidate_ids[i] = -1
            candidate_distances[i] = INF
        for i in range(top_count):
            var value = top_distances[i]
            if value >= candidate_distances[k - 1]:
                continue
            var slot = k - 1
            while slot > 0 and candidate_distances[slot - 1] > value:
                candidate_ids[slot] = candidate_ids[slot - 1]
                candidate_distances[slot] = candidate_distances[slot - 1]
                slot -= 1
            candidate_ids[slot] = top_ids[i]
            candidate_distances[slot] = value
        for i in range(k):
            result_ids[result_offset + i] = candidate_ids[i]
            result_distances[result_offset + i] = Float32(candidate_distances[i])
    else:
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
        for i in range(answer_count):
            result_ids[result_offset + i] = top_ids[i]
            result_distances[result_offset + i] = Float32(top_distances[i])
        for i in range(answer_count, k):
            result_ids[result_offset + i] = -1
            result_distances[result_offset + i] = Float32(INF)


@export("mn_hnsw_search_batch")
def mn_hnsw_search_batch(
    data_addr: Int,
    base_graph_addr: Int,
    upper_graph_addr: Int,
    queries_addr: Int,
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
    upper_degree: Int,
    max_level: Int,
    entry: Int,
    ef: Int,
    k: Int,
    space: Int,
    query_count: Int,
    workers: Int,
) abi("C"):
    if data_addr == 0 or base_graph_addr == 0 or upper_graph_addr == 0 or queries_addr == 0 or result_ids_addr == 0 or result_distances_addr == 0 or visited_addr == 0 or candidate_ids_addr == 0 or candidate_distances_addr == 0 or top_ids_addr == 0 or top_distances_addr == 0:
        return
    if n <= 0 or d <= 0 or degree <= 0 or upper_degree <= 0 or max_level < 0 or ef <= 0 or k <= 0 or k > n or space < 0 or space > 3 or query_count <= 0:
        return
    var data = p(data_addr)
    var base_graph = i32p(base_graph_addr)
    var upper_graph = i32p(upper_graph_addr)
    var queries = p(queries_addr)
    var result_ids = i32p(result_ids_addr)
    var result_distances = f32p(result_distances_addr)
    var visited = u8p(visited_addr)
    var candidate_ids = i32p(candidate_ids_addr)
    var candidate_distances = p(candidate_distances_addr)
    var top_ids = i32p(top_ids_addr)
    var top_distances = p(top_distances_addr)
    if query_count >= MIN_PARALLEL_QUERIES and query_count * n * d >= PARALLEL_WORK and workers != 1:
        var worker_count = min(query_count, MAX_WORKERS)
        if workers > 1:
            worker_count = min(worker_count, workers)
        var data_address = Int(data)
        var base_address = Int(base_graph)
        var upper_address = Int(upper_graph)
        var queries_address = Int(queries)
        var ids_address = Int(result_ids)
        var result_distances_address = Int(result_distances)
        var visited_address = Int(visited)
        var candidate_ids_address = Int(candidate_ids)
        var candidate_distances_address = Int(candidate_distances)
        var top_ids_address = Int(top_ids)
        var top_distances_address = Int(top_distances)

        @parameter
        def work(query_row: Int):
            search_one(
                p(data_address), i32p(base_address), i32p(upper_address), p(queries_address),
                i32p(ids_address), f32p(result_distances_address), u8p(visited_address),
                i32p(candidate_ids_address), p(candidate_distances_address),
                i32p(top_ids_address), p(top_distances_address), n, d, degree,
                upper_degree, max_level, entry, ef, k, space, query_row, query_row,
            )

        parallelize[work](query_count, worker_count)
    else:
        for query_row in range(query_count):
            search_one(
                data, base_graph, upper_graph, queries, result_ids, result_distances,
                visited, candidate_ids, candidate_distances, top_ids, top_distances,
                n, d, degree, upper_degree, max_level, entry, ef, k, space,
                query_row, 0,
            )


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
