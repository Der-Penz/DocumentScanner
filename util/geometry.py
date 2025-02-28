from typing import List, Tuple, Optional
import numpy as np

def line_intersections(lines : List[Tuple[float, float, float]]) -> List[Tuple[Tuple[float, float], Tuple[float, float, float], Tuple[float, float, float]]]:
    """
    calculates all intersection points from the given set of lines

    :param lines: A list of lines in the format: x, y, slope where x and y is a point on the line
    :return: a list of intersections points x y, the intersection angle in degrees and the 2 lines which cross at that point
    """

    intersections = []
    for i, line_a in enumerate(lines):
        for line_b in lines[i+1:]:
            if fuzzy_compare(line_a[2], line_b[2]):
                continue

            intersection = line_intersection(line_a, line_b)
            if intersection:
                point, angle = intersection
                intersections.append((point, angle, line_a, line_b))

    return intersections


def fuzzy_compare(a: float, b: float, epsilon = .001):
    """
    compare if two floats are nearly equal

    :param a: first float
    :param b: second float
    :param epsilon: an error value
    :return: if both floats are nearly equal (defined by epsilon)
    """
    return abs(a-b) < epsilon
            

def line_intersection(line_a : Tuple[float, float, float], line_b: Tuple[float, float, float]) -> Optional[Tuple[Tuple[float, float], float]]:
    """
    calculates the interception point of the two line.

    :param line_a: The first line in the format: x, y, slope where x and y is a point on the line
    :param line_b: The second line in the format: x, y, slope where x and y is a point on the line
    :return: The intersection point x y coords as a Tuple and the intersection angle in degrees or None if lines are parallel or the same
    """
    x_a, y_a, m_a = line_a
    x_b, y_b, m_b = line_b

    if m_a == m_b:
        return None


    b_a = y_a -  m_a * x_a
    b_b = y_b -  m_b * x_b

    x_inter = (b_a - b_b) / (m_b - m_a)
    y_inter = m_a * x_inter + b_a

    if fuzzy_compare(m_a * m_b, -1):
        angle = 90
    else:
        angle =  np.atan((m_a - m_b) / (1 + m_a * m_b))
        angle = np.degrees(angle)


    return (x_inter, y_inter, angle)

def rescale_point(point : Tuple[int, int], cur_shape : Tuple[int, int], original_shape : Tuple[int, int]) -> Tuple[int, int]:
    """
    Undo scaling for a given point in a img shape into the original img shape

    :param point: Points coords in cur_shape as y, x
    :param cur_shape: Size of the image as height, width
    :param original_shape: Size of the original image as height, width
    :return: points coords in the original shape as y, x
    """
    return (int(point[0] * original_shape[0] / cur_shape[0]), int(point[1] * original_shape[1] / cur_shape[1]))


def euclidean_distance(point_a : Tuple[float, ...], point_b : Tuple[float, ...]) -> float:
    """
    calculates the euclidean distance between two points using pythagorean theorem

    :param point_a: the first point
    :param point_b: the second point
    :return distance
    """
    return np.sqrt(np.sum(np.square(np.array(point_a) - np.array(point_b))))