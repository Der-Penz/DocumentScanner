from skimage.transform import rescale
from skimage.morphology import closing
from skimage.feature import canny
from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage.transform import hough_line
from skimage.transform import ProjectiveTransform, warp
import numpy as np
from util.geometry import rescale_point, euclidean_distance
from corner_detection import find_corners_from_hough_space, find_intersections


def detect_document(
    img,
    preferred_min_size=256,
    sigma=1.2,
    num_angles=360,
    max_angle_deviation=20,
    epsilon=0.1,
    threshold=0.2,
    max_retries=20,
    footprint_size=27,
) -> str:
    """
    Detects a document in an image and saves the scanned version.

    :param img: Input image (numpy array).
    :param preferred_min_size: Minimum preferred document size for detection (default: 256).
    :param sigma: Sigma for Gaussian smoothing (default: 1.2).
    :param num_angles: Number of angles for edge detection (default: 360).
    :param max_angle_deviation: Maximum deviation for valid angles (default: 20).
    :param epsilon: Margin percentage for near-edge intersections (default: 0.1).
    :param threshold: Threshold for document detection (default: 0.2).
    :param max_retries: Maximum retries for corner detection (default: 20).
    :param footprint_size: Size of the footprint for morphological operations (default: 27).
    :return: the image
    """
    preprocessed_img = _preprocess_image(
        img,
        preferred_min_size,
        sigma,
        footprint_size,
    )
    edge_map = _edge_detection(preprocessed_img)

    corners = _corner_detection(
        edge_map,
        img.shape,
        num_angles=num_angles,
        max_angle_deviation=max_angle_deviation,
        epsilon=epsilon,
        threshold=threshold,
        max_retries=max_retries,
    )

    if corners is None:
        raise ValueError("No corners found in the image.")

    width, height, orientation = _get_image_out_shape(corners)
    corners = _sort_corners(orientation, corners)

    warped = _warp_image(img, corners, width, height)

    return warped


def _preprocess_image(img, preferred_min_size, sigma, footprint_size):
    min_side = np.min(img.shape[:2])
    scaling_factor = preferred_min_size / min_side
    footprint = [(np.ones((footprint_size, 1)), 1), (np.ones((1, footprint_size)), 1)]

    preprocess_img = rescale(img, scaling_factor, channel_axis=2)
    preprocess_img = rgb2gray(preprocess_img[:, :, :3])
    preprocess_img = closing(preprocess_img, footprint=footprint)
    preprocess_img = gaussian(preprocess_img, sigma=sigma)
    return preprocess_img


def _edge_detection(img):
    return canny(img)


def _corner_detection(
    img,
    original_shape,
    num_angles,
    max_angle_deviation,
    epsilon,
    threshold,
    max_retries,
):
    """
    Detects corners in the image using Hough transform and returns the coordinates of the corners in clockwise order.
    """
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, num_angles, endpoint=False)
    h, h_angles, dists = hough_line(img, theta=tested_angles)

    corners, _ = find_corners_from_hough_space(
        h,
        h_angles,
        dists,
        max_peaks=max_retries,
        img_shape=img.shape,
        epsilon=epsilon,
        threshold_percentage=threshold,
        max_angle_deviation=max_angle_deviation,
        out_shape=original_shape,
    )
    if corners is None:
        return None

    return np.array([corner.point.coords for corner in corners])


def _get_image_out_shape(corners):
    # get the average size of the parallel sides of the document
    horizontal_side_avg = int(
        np.average(
            [
                euclidean_distance(corners[0], corners[1]),
                euclidean_distance(corners[2], corners[3]),
            ]
        )
    )
    vertical_side_avg = int(
        np.average(
            [
                euclidean_distance(corners[0], corners[3]),
                euclidean_distance(corners[1], corners[2]),
            ]
        )
    )

    # define the shortest side as the width and the larger side as the height
    sides = [horizontal_side_avg, vertical_side_avg]
    shortest_idx = np.argmin(sides)
    width = sides[shortest_idx]
    height = max(sides)
    return width, height, shortest_idx == 1


# if the vertical side is the smallest the document is orientated in landscape and needs to be rotated


def _sort_corners(in_landscape_position, corners):
    corner_order = [1, 0, 3, 2] if in_landscape_position else [0, 3, 2, 1]

    return corners[corner_order]


def _warp_image(img, corners, width, height):
    # 0 0 is top left origin
    src = np.array([[0, 0], [0, height], [width, height], [width, 0]])
    dst = corners.copy()

    projectiveTransform = ProjectiveTransform()
    projectiveTransform.estimate(src, dst)
    warped = warp(
        img, projectiveTransform, output_shape=(height, width), mode="constant", cval=1
    )
    warped = (warped * 255).astype(np.uint8)
    return warped
