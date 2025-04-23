from typing import Tuple, Union
import numpy as np

def line_hough_transform(edge_map : np.ndarray, angles : Union[np.ndarray[np.double], None] = None) -> Tuple[np.ndarray, np.ndarray[np.double], np.ndarray[np.double]]:
    """
    Applies a straight line hough transformation on the edge map to find straight lines in the image

    :param edge_map: An binary image where white pixel indicate an edge in the shape (M, N)
    :param angles: Angles (Theta values) at which to compute the transform, in radians. Defaults to a vector of 180 angles evenly spaced in the range [-pi/2, pi/2).

    :returns: A tuple of Hough transform accumulator, used Angles in radians and distance values.
            The indices of a peak in the accumulator can be used to get the theta and rho values from the angles and distances array
    """
    height, width = edge_map.shape

    if angles is None:
        angles = np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False)

    diagonal_length = int(np.ceil((height ** 2 + width ** 2) ** .5))
    distances = np.linspace(-diagonal_length, diagonal_length, diagonal_length*2,)

    sin_cache = np.sin(angles)
    cos_cache = np.cos(angles)

    accumulator = np.zeros(shape=(len(distances), len(angles)))

    for y in range(height):
        for x in range(width):
            value = edge_map[y][x]

            if not value:
                continue

            for i, (sin, cos) in enumerate(zip(sin_cache, cos_cache)):
                rho =  sin * y + cos * x 
                rho = np.round(rho) + diagonal_length
                rho = rho.astype(np.int16)

                accumulator[rho][i] += 1

    
    return accumulator, angles, distances


def hough_lines_peaks(hough_space: np.ndarray, angles : np.ndarray[np.double], dists : np.ndarray[np.double], num_peaks : int = np.inf, threshold = .5, min_distance = 5, min_angle = 1):
    '''
    Find the n peaks in a given hough space by using non maxima suppression to filter out similar lines.

    :param hough_space: the accumulator given from line_hough_transform in shape (rho size, theta size)
    :param angles: The angles used in the hough transform
    :param dists: The distances used in the hough transform
    :param num_peaks: number of max lines that should get return which are greater than threshold
    :param threshold: the minium value a a line needs in the accumulator to be counted as a line
    :param min_distance: the minimum distance a line must be from another to not get suppressed by nms
    :param min_angle: the minimum angle a line must differ from another to not get suppressed by nms
    '''
    line_idx = _non_maxima_suppression(hough_space, angles, dists, num_peaks, threshold, min_distance, min_angle)

    angles_list, dists_list = zip(*[(angles[j], dists[i]) for i, j in line_idx])
    return hough_space, list(angles_list), list(dists_list)


def _non_maxima_suppression(hough_space, angles, dists, max_peaks, threshold, min_distance, min_angle):
    lines = []
    hough_space = hough_space.copy()[:,:-1]
    
    max_value = np.max(hough_space)
    for i in range(max_peaks):
        peak = np.unravel_index(np.argmax(hough_space), hough_space.shape)

        # only take lines over the threshold
        if hough_space[peak] / max_value < threshold:
            break 

        hough_space[peak] = 0
        lines.append(peak)


        lower_distance, upper_distance = dists[peak[0]] - min_distance, dists[peak[0]] + min_distance
        lower_distance, upper_distance = np.searchsorted(dists, lower_distance, side='left'), np.searchsorted(dists, upper_distance, side='left')

        lower_angle, upper_angle = angles[peak[1]] - min_angle, angles[peak[1]] + min_angle
        lower_angle, upper_angle = np.searchsorted(angles, lower_angle, side='left'), np.searchsorted(angles, upper_angle, side='left')

        for i in range(max(lower_distance, 0), min(upper_distance + 1, hough_space.shape[0])):
                for j in range(max(lower_angle, 0), min(upper_angle + 1, hough_space.shape[1])):
                        hough_space[i][j] = 0
    return lines
