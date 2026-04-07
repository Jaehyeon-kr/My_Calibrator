import cv2
import numpy as np

CALIB_FILE = "calibration_result.npz"
VIDEO_PATH = "./data/sample2.mp4"
OUTPUT_PATH = "./data/undistorted.mp4"

def correct_distortion(video_path, calib_file, output_path):
    # Load calibration results
    data = np.load(calib_file)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    print(f"Loaded calibration from '{calib_file}'")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open '{video_path}'")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Optimal new camera matrix (alpha=1: keep all pixels, alpha=0: crop black borders)
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=1
    )

    # Precompute undistort maps for speed
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_matrix, (w, h), cv2.CV_16SC2
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w * 2, h))  # side-by-side

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        # Label
        cv2.putText(frame, "Original", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.putText(undistorted, "Undistorted", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        combined = np.hstack([frame, undistorted])
        out.write(combined)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Saved undistorted video to '{output_path}' ({frame_idx} frames, side-by-side)")

    # Save comparison images at 5 evenly spaced frames
    cap2 = cv2.VideoCapture(video_path)
    total = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    import os
    os.makedirs("comparison_frames", exist_ok=True)
    for i, pos in enumerate([int(total * t) for t in [0.1, 0.3, 0.5, 0.7, 0.9]]):
        cap2.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap2.read()
        if not ret:
            continue
        undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        cv2.putText(frame, "Original", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.putText(undistorted, "Undistorted", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        out_path = f"comparison_frames/comparison_{i+1:02d}_frame{pos}.jpg"
        cv2.imwrite(out_path, np.hstack([frame, undistorted]))
        print(f"Saved {out_path}")
    cap2.release()


if __name__ == "__main__":
    correct_distortion(VIDEO_PATH, CALIB_FILE, OUTPUT_PATH)
