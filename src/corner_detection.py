from typing import List, Optional, Tuple
import numpy as np
from util.geometry import (
    Line,
    Intersection,
    PointLike,
    arg_sort_points_clockwise,
    in_range,
    line_intersections,
)
from skimage.transform import hough_line_peaks


def find_intersections(
    h: np.ndarray,
    angles: np.array,
    dists: np.array,
    img_shape: np.ndarray,
    max_peaks: int,
    threshold: float,
    epsilon: float,
) -> Tuple[List[Intersection], List[Line]]:
    """
    find the intersections from straight lines in the given hough space

    :param h: Hough accumulator array
    :param angles: Array of angles used for the hough accumulator
    :param dists: Array of distances used for the hough accumulator
    :param img_shape: Shape of the image
    :param max_peaks: maximum number of lines to consider in the hough accumulator
    :param threshold: threshold for peak detection in the Hough transform
    :param epsilon: margin as a fraction of image dimensions to allow near-edge intersections

    :returns:
        tuple containing:
            - intersections: List of intersection points
            - lines: List of detected lines
    """
    hough_lines: List[Line] = []
    for _, angle, dist in zip(
        *hough_line_peaks(h, angles, dists, num_peaks=max_peaks, threshold=threshold)
    ):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        slope = np.tan(angle + np.pi / 2)
        hough_lines.append(Line(x0, y0, slope))

    intersections = line_intersections(hough_lines)

    # filter intersection points that are either inside the image or within an epsilon margin defined as a percentage of the image dimensions. The epsilon value allows intersections
    # near the image edges to be included.
    epsilon_y, epsilon_x = epsilon * np.array(img_shape)
    max_y, max_x = img_shape
    max_y, max_x = max_y + epsilon_y, max_x + epsilon_x
    intersections = [
        inter
        for inter in intersections
        if -epsilon_x <= inter.point.x <= max_x and -epsilon_y <= inter.point.y <= max_y
    ]

    return intersections, hough_lines


def detect_corners(
    intersections: List[Intersection], max_angle_deviation: float
) -> Optional[List[Intersection]]:
    """
    Detect the corners of a document based on the intersection points of lines.
    Intersections are ranked by several criteria, to determine the best candidates for document corners.

    :param intersections: List of intersection points
    :param max_angle_deviation: Maximum allowed angle deviation from 90 degrees to consider a point as a corner

    :return: List of the 4 detected corners or None if not enough corners are found
    """

    intersections = list(
        filter(
            lambda inter: in_range(
                abs(inter.angle),
                90 - max_angle_deviation,
                90 + max_angle_deviation,
            ),
            intersections,
        )
    )

    return intersections if len(intersections) >= 4 else None


def find_corners_from_hough_space(
    h,
    theta,
    d,
    max_peaks: int,
    img_shape: PointLike,
    epsilon: float,
    threshold_percentage: float,
):
    """
    Find the corners of a document in the Hough space. Tries to find the best 4 corners by starting with only 4 lines and increasing the number of lines until enough corners are found.

    :param h: Hough accumulator array
    :param theta: Array of angles used for the Hough transform
    :param d: Array of distances used for the Hough transform
    :param max_peaks: Maximum number of lines to consider in the Hough accumulator
    :param img_shape: Shape of the image
    :param epsilon: Margin as a fraction of image dimensions to allow near-edge intersections
    :param threshold_percentage: Percentage of the maximum value in the Hough accumulator to use as a threshold for peak detection

    :return: Tuple containing the detected corners and the lines. If no corners are found, returns None and the lines.
    """
    peaks = 4

    while True:
        intersections, lines = find_intersections(
            h,
            theta,
            d,
            img_shape=img_shape,
            max_peaks=peaks,
            threshold=threshold_percentage * max([max(a) for a in h]),
            epsilon=epsilon,
        )

        corners = detect_corners(intersections, max_angle_deviation=20)

        if corners is None:
            if max_peaks == peaks:
                return None, lines
            peaks += 1
            continue

        order = arg_sort_points_clockwise([corner.point for corner in corners])
        return np.array(corners)[order], lines
