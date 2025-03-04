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
