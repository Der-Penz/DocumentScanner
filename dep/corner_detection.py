import numpy as np
from util.geometry import line_intersections, rescale_point
from skimage.transform import hough_line_peaks

def find_corners(img : np.ndarray, h : np.ndarray, angles : np.array, dists : np.array, max_peaks : int, threshold : float, epsilon: float, max_angle_deviation: float):
    """
    find the corners from straight lines in the given hough space 

    :param img: Input image
    :param h: Hough accumulator array
    :param angles: Array of angles used for the hough accumulator
    :param dists: Array of distances used for the hough accumulator
    :param max_peaks: maximum number of lines to consider in the hough accumulator
    :param threshold: threshold for peak detection in the Hough transform
    :param epsilon: margin as a fraction of image dimensions to allow near-edge intersections
    :param max_angle_deviation: maximum allowable deviation from 90° for valid corner angles

    :returns:
        tuple of
            - corners: List of detected corner points (x, y)  
            - angles: List of angles at detected corners  
            - hough_lines: List of detected Hough lines as (x0, y0, slope)  
    """
    hough_lines = []
    for _, angle, dist in zip(*hough_line_peaks(h, angles, dists, num_peaks=max_peaks, threshold=threshold)):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        slope = np.tan(angle + np.pi / 2)
        hough_lines.append((x0, y0, slope))

    intersections = line_intersections(hough_lines)
    
    # filter intersection points that are either inside the image or within an epsilon margin defined as a percentage of the image dimensions. The epsilon value allows intersections
    # near the image edges to be included.
    epsilon_y, epsilon_x = epsilon * np.array(img.shape)
    max_y, max_x = img.shape
    max_y, max_x = max_y + epsilon_y, max_x + epsilon_x
    intersections = [inter for inter in intersections if -epsilon_x <= inter[0][0] <= max_x and -epsilon_y <= inter[0][1] <= max_y ]

    #filter out unlikely document corners (angle far off from 90°)
    intersections = [inter for inter in intersections if np.abs(90 - np.abs(inter[1])) < max_angle_deviation]
    if len(intersections) == 0:
        return [], [], hough_lines

    corners, angle = list(zip(*[(point, angle) for point, angle, *_ in intersections]))
    return corners, angle, hough_lines