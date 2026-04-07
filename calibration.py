import cv2
import numpy as np

# Chessboard inner corners (columns x rows)
# Standard printed chessboard: 9x6 inner corners → adjust if detection fails
CHESSBOARD_SIZE = (6, 4)
SQUARE_SIZE_X = 1.0
SQUARE_SIZE_Y = 1.0

VIDEO_PATH = "./data/sample2.mp4"


def calibrate_camera(video_path, chessboard_size=CHESSBOARD_SIZE,
                     square_size_x=SQUARE_SIZE_X, square_size_y=SQUARE_SIZE_Y):
    # Prepare object points with per-axis square size for non-square grids
    cols, rows = chessboard_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, 0] = np.tile(np.arange(cols), rows) * square_size_x   # x
    objp[:, 1] = np.repeat(np.arange(rows), cols) * square_size_y  # y

    obj_points = []  # 3D points in real world
    img_points = []  # 2D points in image plane

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video '{video_path}'")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps:.1f}")

    # Sample one frame per second for diverse viewpoints
    sample_interval = max(1, int(fps))
    frame_idx = 0
    detected_count = 0
    gray = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            found, corners = False, None
            for img in [gray, clahe.apply(gray),
                        cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)]:
                found, corners = cv2.findChessboardCorners(img, chessboard_size, flags)
                if found:
                    if img.shape[0] == gray.shape[0] * 2:
                        corners = corners / 2.0
                    break

            if found:
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria
                )
                obj_points.append(objp)
                img_points.append(corners_refined)
                detected_count += 1
                print(f"  Frame {frame_idx:5d}: chessboard detected (total: {detected_count})")

        frame_idx += 1

    cap.release()
    print(f"\nDetected chessboard in {detected_count} frames.")

    if detected_count == 0:
        print("No chessboard detected. Check CHESSBOARD_SIZE setting.")
        return

    if detected_count < 10:
        print("Warning: fewer than 10 detections — calibration may be inaccurate.")

    # Run calibration
    h, w = gray.shape
    print(f"Running calibration on {w}x{h} images with {detected_count} frames...")

    rmse, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None
    )

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    k1, k2, p1, p2, k3 = dist_coeffs.ravel()[:5]

    print("\n========== Camera Calibration Results ==========")
    print(f"  fx = {fx:.4f}")
    print(f"  fy = {fy:.4f}")
    print(f"  cx = {cx:.4f}")
    print(f"  cy = {cy:.4f}")
    print(f"  k1 = {k1:.6f}")
    print(f"  k2 = {k2:.6f}")
    print(f"  p1 = {p1:.6f}")
    print(f"  p2 = {p2:.6f}")
    print(f"  k3 = {k3:.6f}")
    print(f"  RMSE (reprojection error) = {rmse:.4f} pixels")
    print("================================================\n")

    # Save results for use in distortion correction
    np.savez("calibration_result.npz",
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             rmse=rmse)
    print("Calibration data saved to 'calibration_result.npz'")

    return camera_matrix, dist_coeffs, rmse


if __name__ == "__main__":
    calibrate_camera(VIDEO_PATH, square_size_x=SQUARE_SIZE_X, square_size_y=SQUARE_SIZE_Y)