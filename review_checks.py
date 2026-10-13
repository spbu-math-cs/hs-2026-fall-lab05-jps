"""Correctness and performance checks for the 8-connected JPS lab."""

import io
import math
import time
from contextlib import redirect_stdout
from pathlib import Path

from jps_utils.map import Map
from jps_utils.utils import make_path, read_task_from_file


def _run_quiet(search, *args):
    with redirect_stdout(io.StringIO()):
        return search(*args)


def _validate(result, task_map, start, goal, expected_length):
    assert isinstance(result, tuple) and len(result) == 6, "Search must return six values."
    found, end, steps, tree_size, opened, expanded = result
    assert found and end is not None, "A path should be found."
    assert isinstance(steps, int) and steps >= 0
    assert isinstance(tree_size, int) and tree_size >= 0
    assert opened is not None and expanded is not None
    path, length = make_path(end)
    assert (path[0].i, path[0].j) == start and (path[-1].i, path[-1].j) == goal
    assert math.isclose(length, expected_length, rel_tol=1e-9), (
        f"Expected length {expected_length}, got {length}.")
    measured = 0.0
    for first, second in zip(path, path[1:]):
        di, dj = abs(first.i-second.i), abs(first.j-second.j)
        assert max(di, dj) == 1 and di + dj > 0, "Path has an illegal move."
        assert task_map.traversable(second.i, second.j), "Path crosses an obstacle."
        measured += math.sqrt(2) if di and dj else 1.0
    assert math.isclose(length, measured, rel_tol=1e-9), "Path and reported length differ."
    return length


def review_jps(MapClass, Node, SearchTreePQD, heuristic, astar, jps):
    import numpy as np
    maps = [
        (np.zeros((1, 1), dtype=np.int8), (0, 0), (0, 0)),
        (np.zeros((12, 15), dtype=np.int8), (0, 0), (11, 14)),
        (np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int8), (0, 0), (0, 2)),
    ]
    for cells, start, goal in maps:
        task_map = MapClass(cells)
        baseline = _run_quiet(astar, task_map, *start, *goal, heuristic, SearchTreePQD)
        candidate = _run_quiet(jps, task_map, *start, *goal, heuristic, SearchTreePQD)
        assert baseline[0] == candidate[0]
        if baseline[0]:
            _validate(candidate, task_map, start, goal, baseline[1].g)
    blocked = MapClass(np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.int8))
    result = _run_quiet(jps, blocked, 0, 0, 0, 2, heuristic, SearchTreePQD)
    assert result[0] is False and result[1] is None and result[4] is None
    open_map = MapClass(np.zeros((1000, 1), dtype=np.int8))
    a_result = _run_quiet(astar, open_map, 0, 0, 999, 0, heuristic, SearchTreePQD)
    j_result = _run_quiet(jps, open_map, 0, 0, 999, 0, heuristic, SearchTreePQD)
    assert math.isclose(j_result[1].g, a_result[1].g) and j_result[2] < a_result[2]
    print("8-connected JPS assertion checks passed.")


def compare_on_tasks(astar, jps, data_path, number_of_tasks, heuristic, search_tree):
    rows = []
    totals = {"astar_time": 0.0, "jps_time": 0.0, "astar_steps": 0, "jps_steps": 0}
    for index in range(number_of_tasks):
        cells, si, sj, gi, gj, expected = read_task_from_file(Path(data_path) / f"{index}.map")
        task_map = Map(cells)
        started = time.perf_counter()
        a_result = _run_quiet(astar, task_map, si, sj, gi, gj, heuristic, search_tree)
        a_time = time.perf_counter() - started
        started = time.perf_counter()
        j_result = _run_quiet(jps, task_map, si, sj, gi, gj, heuristic, search_tree)
        j_time = time.perf_counter() - started
        _validate(a_result, task_map, (si, sj), (gi, gj), expected)
        _validate(j_result, task_map, (si, sj), (gi, gj), expected)
        rows.append((index, a_time, j_time, a_result[2], j_result[2]))
        totals["astar_time"] += a_time
        totals["jps_time"] += j_time
        totals["astar_steps"] += a_result[2]
        totals["jps_steps"] += j_result[2]
        print(f"Task {index}: A* {a_time:.4f}s/{a_result[2]} steps; "
              f"JPS {j_time:.4f}s/{j_result[2]} steps")
    assert totals["jps_steps"] < totals["astar_steps"], "JPS must expand fewer nodes overall."
    assert totals["jps_time"] < totals["astar_time"], "JPS must be faster overall."
    assert rows[-1][2] < rows[-1][1], "JPS must be faster on the performance task."
    print(f"Total speedup: {totals['astar_time'] / totals['jps_time']:.2f}x")
    return {"tasks": rows, **totals}
