from typing import List, Tuple, Union, Iterable, Optional
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from jps_utils.map import Map
from jps_utils.node import BaseNode as Node


def convert_string_to_cells(cell_str: str) -> npt.NDArray:
    """
    Converts a string representation of a grid map, with '#' for obstacles
    and '.' for free cells, into a binary matrix.

    Parameters
    ----------
    cell_str : str
        String containing grid map information ('#' for obstacles and '.' for free cells).

    Returns
    ----------
    cells : np.ndarray
        Binary matrix representing the grid map.
    """
    lines = cell_str.replace(" ", "").split("\n")

    cells = np.array(
        [[1 if char == "#" else 0 for char in line] for line in lines if line],
        dtype=np.int8,
    )
    return cells


def compute_cost(i1: int, j1: int, i2: int, j2: int) -> Union[int, float]:
    """Return the octile cost of a straight or 45-degree diagonal segment.

    Parameters
    ----------
    i1, j1 : int, int
        Coordinates of the first cell.
    i2, j2 : int, int
        Coordinates of the second cell.

    Returns
    -------
    int | float
        Cardinal distance for a straight segment or ``sqrt(2)`` per diagonal step.

    Raises
    ------
    ValueError
        If the cells do not form a straight or 45-degree diagonal segment.
    """
    di, dj = abs(i1 - i2), abs(j1 - j2)
    if di == 0 or dj == 0:
        return di + dj
    if di == dj:
        return di * np.sqrt(2)
    raise ValueError("A JPS edge must be straight or a 45-degree diagonal segment.")


def read_task_from_file(
    path: str,
) -> Tuple[npt.NDArray, int, int, int, int, float]:
    """
    Reads map, start/goal positions, and true value of path length 
    between the given start and goal from a file.

    Parameters
    ----------
    path : str
        Path to a file with the pathfinding problem instance.

    Returns
    -------
    cells : npt.NDArray
        Matrix of grid map cells.
    start_i, start_j : int, int
        Position of the start cell in the grid.
    goal_i, goal_j : int, int
        Position of the goal cell in the grid.
    length : float
        True value of the path's length between the given start and goal.
    """
    with open(path) as tasks_file:
        height = int(next(tasks_file))
        width = int(next(tasks_file))

        # Read the map section
        map_lines = [next(tasks_file) for _ in range(height)]
        map_str = "".join(map_lines)
        cells = convert_string_to_cells(map_str)

        start_i = int(next(tasks_file))
        start_j = int(next(tasks_file))
        goal_i = int(next(tasks_file))
        goal_j = int(next(tasks_file))
        length = float(next(tasks_file))

    return cells, start_i, start_j, goal_i, goal_j, length


def make_path(goal: Node) -> Tuple[List[Node], Union[float, int]]:
    """
    Creates a path by tracing parent pointers from the goal node to the start node.
    It also returns the path's length.

    Parameters
    ----------
    goal : Node
        Pointer to the goal node in the search tree.

    Returns
    -------
    Tuple[List[Node], float]
        Path and its length.
    """
    length = goal.g
    jump_points = []
    current = goal
    while current is not None:
        jump_points.append(current)
        current = current.parent
    jump_points.reverse()

    # A* parents are adjacent. JPS parents are jump points, so interpolate each
    # horizontal or vertical segment for validation and for the standard drawing.
    path = []
    for start, end in zip(jump_points, jump_points[1:]):
        if not path:
            path.append(start)
        di = (end.i > start.i) - (end.i < start.i)
        dj = (end.j > start.j) - (end.j < start.j)
        i, j = start.i, start.j
        while (i, j) != (end.i, end.j):
            i, j = i + di, j + dj
            path.append(Node(i, j))
    if not path and jump_points:
        path.append(jump_points[0])
    return path, length


def draw_rectangle(draw, node, scale, color):
    """Draws a rectangle on the grid, given the node and the desired color."""
    draw.rectangle(
        (
            node.j * scale,
            node.i * scale,
            (node.j + 1) * scale - 1,
            (node.i + 1) * scale - 1,
        ),
        fill=color,
        width=0,
    )


def draw(
    grid_map: Map,
    start: Optional[Node] = None,
    goal: Optional[Node] = None,
    path: Optional[Iterable[Node]] = None,
    nodes_discovered: Optional[Iterable[Node]] = None,
    nodes_expanded: Optional[Iterable[Node]] = None,
    nodes_reexpanded: Optional[Iterable[Node]] = None,
):
    """
    Visualizes the environment, path, and expanded/not yet expanded/re-expanded nodes.

    Re-expansions don't occur in Dijkstra's algorithm, but are included for potential future use.

    Parameters
    ----------
    grid_map : Map
        Environment represented as a grid.
    start, goal : Optional[Node], Optional[Node]
        Nodes for the start and goal positions of the agent.
    path : Optional[Iterable[Node]]
        Sequence of nodes representing the path between start and goal.
    nodes_discovered : Optional[Iterable[Node]]
        Nodes that were discovered during the search process.
    nodes_expanded : Optional[Iterable[Node]]
        Nodes that were expanded during the search process.
    nodes_reexpanded : Optional[Iterable[Node]]
        Nodes that were re-expanded during the search process.
    """
    scale = 5
    height, width = grid_map.get_size()
    im = Image.new("RGB", (width * scale, height * scale), color="white")
    draw = ImageDraw.Draw(im)

    # Draw blocked cells
    for i in range(height):
        for j in range(width):
            if not grid_map.traversable(i, j):
                draw.rectangle(
                    (j * scale, i * scale, (j + 1) * scale - 1, (i + 1) * scale - 1),
                    fill=(70, 80, 80),
                )

    node_colors = [
        (nodes_discovered, (213, 219, 219)),
        (nodes_expanded, (131, 145, 146)),
        (nodes_reexpanded, (255, 145, 146)),
    ]

    for nodes, color in node_colors:
        if nodes is not None:
            for node in nodes:
                draw_rectangle(draw, node, scale, color)

    if path is not None:
        for step in path:
            color = (
                (52, 152, 219)
                if grid_map.traversable(step.i, step.j)
                else (230, 126, 34)
            )
            draw_rectangle(draw, step, scale, color)

    if start is not None and grid_map.traversable(start.i, start.j):
        draw_rectangle(draw, start, scale, (40, 180, 99))

    if goal is not None and grid_map.traversable(goal.i, goal.j):
        draw_rectangle(draw, goal, scale, (231, 76, 60))

    _, ax = plt.subplots(dpi=150)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.imshow(np.asarray(im))
    plt.show()
