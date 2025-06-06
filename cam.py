import argparse
import cv2
import tempfile
import threading

import numpy as np

from detection import (
    detect_document,
    _edge_detection,
    _corner_detection,
    _preprocess_image,
)

corners = None
lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect a document in an image from the given video stream."
    )

    parser.add_argument("url", type=str, help="HTTP URL of the camera stream.")
    parser.add_argument(
        "--preferred-min-size",
        type=int,
        default=256,
        help="Minimum preferred document size (default: 256).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.2,
        help="Sigma for Gaussian smoothing (default: 1.2).",
    )
    parser.add_argument(
        "--num-angles",
        type=int,
        default=360,
        help="Number of angles for edge detection (default: 360).",
    )
    parser.add_argument(
        "--max-angle-deviation",
        type=int,
        default=20,
        help="Maximum deviation for valid angles (default: 20).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Margin percentage for near-edge intersections (default: 0.1).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Threshold for document detection (default: 0.2).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=20,
        help="Maximum retries for corner detection (default: 20).",
    )

    parser.add_argument(
        "--corner-calc-freq",
        type=int,
        default=20,
        help="Frequency of corner calculation in frames (default: every 20th frame).",
    )
    parser.add_argument(
        "--footprint-size",
        type=int,
        default=27,
        help="Size of the footprint for morphological operations (default: 27).",
    )

    args = parser.parse_args()
    return args


def find_corners(frame, args):
    global corners

    preprocessed_img = _preprocess_image(
        frame,
        256,
        1.2,
        footprint_size=args.footprint_size,
    )
    edge_map = _edge_detection(preprocessed_img)
    new_corners = _corner_detection(
        edge_map,
        frame.shape,
        num_angles=args.num_angles,
        max_angle_deviation=args.max_angle_deviation,
        epsilon=args.epsilon,
        threshold=args.threshold,
        max_retries=args.max_retries,
    )
    if new_corners is not None:
        with lock:
            corners = new_corners


def process_frame(frame, args):
    try:
        detected_image = detect_document(
            img=frame,
            preferred_min_size=args.preferred_min_size,
            sigma=args.sigma,
            num_angles=args.num_angles,
            max_angle_deviation=args.max_angle_deviation,
            epsilon=args.epsilon,
            threshold=args.threshold,
            max_retries=args.max_retries,
        )

        return detected_image
    except Exception as e:
        print(f"Error during document detection:")
        raise e


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    h, w = image.shape[:2]
    if width is None and height is None:
        return image
    if width is not None:
        scale = width / float(w)
        dim = (width, int(h * scale))
    else:
        scale = height / float(h)
        dim = (int(w * scale), height)
    return cv2.resize(image, dim, interpolation=inter)


if __name__ == "__main__":

    args = parse_args()

    if not args.url.startswith("http://") and not args.url.startswith("https://"):
        print("Invalid URL. Please provide a valid HTTP/HTTPS URL.")
        exit(1)
    try:
        cap = cv2.VideoCapture(args.url)
    except Exception as e:
        print(f"Error: {e}")
        print("Failed to open camera stream. Please check the URL.")
        exit(1)

    i = 0
    corners = None
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame.")
            break

        frame = resize_with_aspect_ratio(frame, width=1080)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            cv2.destroyAllWindows()
            break

        if i % args.corner_calc_freq == 0:
            i -= args.corner_calc_freq
            thread = threading.Thread(target=find_corners, args=(frame.copy(), args))
            thread.start()

        i += 1

        with lock:
            current_corners = corners.copy() if corners is not None else None

        if current_corners is not None:
            copy_frame = frame.copy()
            for corner in current_corners:
                x, y = corner
                cv2.circle(
                    copy_frame,
                    (int(x), int(y)),
                    radius=5,
                    color=(0, 255, 0),
                    thickness=-1,
                )
            cv2.polylines(
                copy_frame,
                [np.array(current_corners, dtype=np.int32)],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2,
            )
            cv2.imshow("Stream", copy_frame)
        else:
            cv2.imshow("Stream", frame)

        if key == ord("d"):

            img = process_frame(frame, args)

            if img is not None:
                cv2.imshow("Detected Document", img)
                while True:
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        cv2.destroyWindow("Detected Document")
                        break
                    if cv2.waitKey(1) & 0xFF == ord("s"):
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".jpg"
                        ) as temp_file:
                            temp_file.write(img)
                            print(f"Image saved to {temp_file.name}")
                        cv2.destroyWindow("Detected Document")
                        break

    cap.release()
