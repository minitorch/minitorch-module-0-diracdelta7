import math
import random
from dataclasses import dataclass
from typing import Callable


Point = tuple[float, float]


def make_pts(N: int) -> list[Point]:
    """Generate a list of random 2D points in the unit square."""
    X: list[Point] = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    """Store a synthetic dataset of points and binary labels."""

    N: int
    X: list[Point]
    y: list[int]


def simple(N: int) -> Graph:
    """Label points by whether their x-coordinate is less than 0.5."""
    X = make_pts(N)
    y: list[int] = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Label points by whether they fall below the diagonal threshold."""
    X = make_pts(N)
    y: list[int] = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Label points by whether their x-coordinate lies near either edge."""
    X = make_pts(N)
    y: list[int] = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Label points by whether they lie in opposite quadrants."""
    X = make_pts(N)
    y: list[int] = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 and x_2 > 0.5 or x_1 > 0.5 and x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Label points by whether they lie outside a centered circle."""
    X = make_pts(N)
    y: list[int] = []
    for x_1, x_2 in X:
        x1, x2 = x_1 - 0.5, x_2 - 0.5
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generate a two-class spiral dataset."""

    def x(t: float) -> float:
        """Map a spiral parameter to its x-offset."""
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        """Map a spiral parameter to its y-offset."""
        return t * math.sin(t) / 20.0

    X: list[Point] = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets: dict[str, Callable[[int], Graph]] = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
