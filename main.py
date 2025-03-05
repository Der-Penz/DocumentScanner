import argparse
from dep.DocumentScanner import detect_document

def main():
    parser = argparse.ArgumentParser(description="Detect a document in an image.")
    
    parser.add_argument("img_path", type=str, help="Path to the input image.")
    parser.add_argument("--preferred_min_size", type=int, default=256, help="Minimum preferred document size (default: 256).")
    parser.add_argument("--sigma", type=float, default=1.2, help="Sigma for Gaussian smoothing (default: 1.2).")
    parser.add_argument("--num_angles", type=int, default=360, help="Number of angles for edge detection (default: 360).")
    parser.add_argument("--max_angle_deviation", type=int, default=20, help="Maximum deviation for valid angles (default: 20).")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Margin percentage for near-edge intersections (default: 0.1).")
    parser.add_argument("--threshold", type=float, default=0.2, help="Threshold for document detection (default: 0.2).")
    parser.add_argument("--max_retries", type=int, default=20, help="Maximum retries for corner detection (default: 20).")
    parser.add_argument("--out", type=str, default="./out", help="Path to save the output image (optional).")
    
    args = parser.parse_args()

    detect_document(
        img_path=args.img_path,
        preferred_min_size=args.preferred_min_size,
        sigma=args.sigma,
        num_angles=args.num_angles,
        max_angle_deviation=args.max_angle_deviation,
        epsilon=args.epsilon,
        threshold=args.threshold,
        max_retries=args.max_retries,
        out=args.out
    )

if __name__ == "__main__":
    main()