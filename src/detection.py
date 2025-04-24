from os import path
from skimage.io import imread, imsave
from skimage.transform import rescale
from skimage.morphology import closing
from skimage.feature import canny
from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage.transform import hough_line
from skimage.transform import ProjectiveTransform, warp
import numpy as np
from util.geometry import rescale_point, euclidean_distance
from corner_detection import find_corners


def detect_document(
    img,
    preferred_min_size=256,
    sigma=1.2,
    num_angles=360,
    max_angle_deviation=20,
    epsilon=0.1,
    threshold=0.2,
    max_retries=20,
) -> str:
    """
    Detects a document in an image and saves the scanned version.

    :param img: Input image (numpy array).
    :param preferred_min_size: Minimum preferred document size (default: 256).
    :param sigma: Sigma for Gaussian smoothing (default: 1.2).
    :param num_angles: Number of angles for edge detection (default: 360).
    :param max_angle_deviation: Maximum deviation for valid angles (default: 20).
    :param epsilon: Margin percentage for near-edge intersections (default: 0.1).
    :param threshold: Threshold for document detection (default: 0.2).
    :param max_retries: Maximum retries for corner detection (default: 20).
    :return: the image
    """
    preprocessed_img = _preprocess_image(
        img,
        preferred_min_size,
        sigma,
        footprint=[(np.ones((27, 1)), 1), (np.ones((1, 27)), 1)],
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

    width, height, orientation = _get_image_out_shape(corners)
    corners = _sort_corners(orientation, corners)

    warped = _warp_image(img, corners, width, height)

    return warped


def _preprocess_image(img, preferred_min_size, sigma, footprint):
    min_side = np.min(img.shape[:2])
    scaling_factor = preferred_min_size / min_side

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
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, num_angles, endpoint=False)
    h, h_angles, dists = hough_line(img, theta=tested_angles)

    max_peaks = max_retries
    peaks = 4
    while True:
        corners, angles, _ = find_corners(
            img,
            h,
            h_angles,
            dists,
            max_peaks=peaks,
            threshold=threshold * max([max(a) for a in h]),
            epsilon=epsilon,
            max_angle_deviation=max_angle_deviation,
        )
        corners = [
            rescale_point(corner, img.shape, original_shape[:-1]) for corner in corners
        ]
        if len(corners) >= 4:
            break

        if max_peaks == peaks:
            break
        peaks += 1

    if len(corners) < 4:
        raise Exception("Document could not be detected")

    # find the best 4 corners (closest to 90° angles)
    if len(corners) > 4:
        ord = np.argsort(np.abs(90 - np.abs(angles)))
        corners = np.array(corners)[ord][:4]
        angles = np.array(angles)[ord][:4]

    corners = np.array(corners)
    centroid = np.mean(corners, axis=0)


    # sort the corners in counter clockwise order starting form the 3. Quadrant
    # since matplotlib coords start in the topleft corner and the y axis is positive downwards the Quadrants are mirrored horizontally
    centroid_angles = np.arctan2(
        corners[:, 1] - centroid[1], corners[:, 0] - centroid[0]
    )
    corners = np.array(corners)[np.argsort(centroid_angles)]
    angles = np.array(angles)[np.argsort(centroid_angles)]

    return corners


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
