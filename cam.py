import argparse
import os
import time
import cv2
import tempfile

from detection import detect_document

parser = argparse.ArgumentParser(
    description="Detect a document in an image from the given video stream."
)

parser.add_argument("url", type=str, help="HTTP URL of the camera stream.")
parser.add_argument(
    "--preferred_min_size",
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
    "--num_angles",
    type=int,
    default=360,
    help="Number of angles for edge detection (default: 360).",
)
parser.add_argument(
    "--max_angle_deviation",
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
    "--max_retries",
    type=int,
    default=20,
    help="Maximum retries for corner detection (default: 20).",
)

args = parser.parse_args()


def process_frame(frame):
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
        return None


try:
    cap = cv2.VideoCapture(args.url)
except Exception as e:
    print(f"Error: {e}")
    print("Failed to open camera stream. Please check the URL.")
    exit(1)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        cv2.destroyAllWindows()
        break

    elif key == ord("d"):

        img = process_frame(frame)

        if img is not None:
            cv2.imshow("Detected Document", img)
            while True:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyWindow("Detected Document")
                    break
                if cv2.waitKey(1) & 0xFF == ord("s"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                        temp_file.write(img)
                        print(f"Image saved to {temp_file.name}")
                    cv2.destroyWindow("Detected Document")
                    break

cap.release()
