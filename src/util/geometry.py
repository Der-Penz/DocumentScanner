from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import itertools


@dataclass
class Point:
    x: float
    y: float

    @property
    def coords(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def rescale(self, cur_shape: PointLike, original_shape: PointLike):
        """
        Undo scaling for the given point in a img shape into the original img shape
        This will modify the point in place

        :param cur_shape: Size of the image
        :param original_shape: Size of the original image
        """
        point = rescale_point(self, cur_shape, original_shape)
        self.x = point[0]
        self.y = point[1]

    def __repr__(self) -> str:
        return f"({self.x}, {self.y})"

    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            raise IndexError("Index out of range for Point. Use 0 for x and 1 for y.")

    def __eq__(self, value):
        if isinstance(value, Point):
            return fuzzy_compare(self.x, value.x) and fuzzy_compare(self.y, value.y)
        return False


type PointLike = Tuple[float, float] | Tuple[int, int] | Point


@dataclass(frozen=True)
class Intersection:
    point: Point
    angle: float
    line_a: Line
    line_b: Line

    def close_to(self, other: Intersection) -> bool:
        return fuzzy_compare(self.point.x, other.point.x) and fuzzy_compare(
            self.point.y, other.point.y
        )

    def __repr__(self) -> str:
        return f"{self.point} | {self.angle}°"


@dataclass(frozen=True)
class Line:
    x: float
    y: float
    slope: float

    @staticmethod
    def from_points(point_a: Point, point_b: Point) -> Line:
        if point_a.x == point_b.x:
            slope = np.inf
        else:
            slope = (point_b.y - point_a.y) / (point_b.x - point_a.x)

        return Line(point_a.x, point_a.y, slope)

    @staticmethod
    def from_m_b(m: float, b: float) -> Line:
        if m == np.inf:
            return Line(0, b, m)
        x = 0
        y = b
        return Line(x, y, m)

    @property
    def b(self) -> float:
        return self.y - self.slope * self.x

    @property
    def m(self) -> float:
        return self.slope

    def __repr__(self) -> str:
        return f"y = {self.slope} * x + {self.b}"

    def intersection(self, other: Line) -> Optional[Intersection]:
        if fuzzy_compare(self.slope, other.slope) or self == other:
            return None

        x_inter = (self.b - other.b) / (other.m - self.m)
        y_inter = self.m * x_inter + self.b

        # Verify y using line B
        y_inter_b = other.m * x_inter + other.b

        # if the y values are not equal, we are likely dealing with a numerical error, so we take the more stable result
        if not fuzzy_compare(y_inter, y_inter_b):
            idx = np.argmin([abs(self.m) + abs(self.b), abs(other.m) + abs(other.b)])
            y_inter = [y_inter, y_inter_b][idx]

        if fuzzy_compare(self.m * other.m, -1):
            angle = 90
        else:
            angle = np.atan((self.m - other.m) / (1 + self.m * other.m))
            angle = np.degrees(angle)

        return Intersection(Point(x_inter, y_inter), angle, self, other)

    def __eq__(self, value):
        if isinstance(value, Line):
            return fuzzy_compare(self.m, value.m) and fuzzy_compare(self.b, value.b)
        return False


def line_intersections(
    lines: List[Line],
) -> List[Intersection]:
    """
    calculates all intersection points from the given set of lines

    :param lines: A list of lines in the format: x, y, slope where x and y is a point on the line
    :return: a list of intersections points x y, the intersection angle in degrees and the indices of the 2 lines which cross at that point
    """
    intersections = []
    for line_a, line_b in itertools.combinations(lines, 2):
        intersection = line_a.intersection(line_b)
        if intersection:
            intersections.append(intersection)

    return intersections


def fuzzy_compare(a: float, b: float, epsilon=0.001):
    """
    compare if two floats are nearly equal

    :param a: first float
    :param b: second float
    :param epsilon: an error value
    :return: if both floats are nearly equal (defined by epsilon)
    """
    return abs(a - b) < epsilon


def in_range(value: float, min_value: float, max_value: float) -> bool:
    """
    check if a value is in a given range

    :param value: the value to check
    :param min_value: the minimum value of the range
    :param max_value: the maximum value of the range
    :return: True if the value is in the range, False otherwise
    """
    return min_value <= value <= max_value


def rescale_point(
    point: PointLike, cur_shape: PointLike, original_shape: PointLike
) -> PointLike:
    """
    Undo scaling for a given point in a img shape into the original img shape

    :param point: Points coords in cur_shape
    :param cur_shape: Size of the image
    :param original_shape: Size of the original image
    :return: points coords in the original shape
    """
    return (
        int(point[0] * original_shape[0] / cur_shape[0]),
        int(point[1] * original_shape[1] / cur_shape[1]),
    )


def euclidean_distance(point_a: Tuple[float, ...], point_b: Tuple[float, ...]) -> float:
    """
    calculates the euclidean distance between two points using pythagorean theorem

    :param point_a: the first point
    :param point_b: the second point
    :return distance
    """
    return np.sqrt(np.sum(np.square(np.array(point_a) - np.array(point_b))))


def arg_sort_points_clockwise(points: List[PointLike]) -> np.NDArray[np.integer]:
    """
    Sorts a list of points in clockwise order around their centroid.

    :param points: List of Point objects to be sorted.
    :return: Indices of the points sorted in clockwise order.
    """
    coords = np.array([[p[0], p[1]] for p in points])
    centroid = np.mean(coords, axis=0)

    angles = np.arctan2(coords[:, 1] - centroid[1], coords[:, 0] - centroid[0])
    return np.argsort(angles)
