"""
Debug script: try multiple chessboard sizes + preprocessing options.
Saves annotated images so you can visually verify detection.
"""
import cv2
import numpy as np
import os

VIDEO_PATH = "./data/sample2.mp4"
OUTPUT_DIR = "./debug_frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CANDIDATES = [
    (7, 7), (7, 3), (8, 4), (6, 3), (5, 3), (4, 3),
    (7, 4), (6, 4), (5, 4), (9, 6), (8, 5),
]

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Extract ~15 frames spread across the video
sample_frames = []
for i in range(15):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * total / 15))
    ret, frame = cap.read()
    if ret:
        sample_frames.append(frame)
cap.release()

print(f"Extracted {len(sample_frames)} frames\n")

# Save raw frames
for i, f in enumerate(sample_frames):
    cv2.imwrite(f"{OUTPUT_DIR}/raw_{i:02d}.jpg", f)

def try_detect(gray, size):
    """Try detection with several preprocessing options, return corners or None."""
    # CLAHE: adaptive histogram equalization (better than global equalizeHist for uneven lighting)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    attempts = [
        gray,
        cv2.equalizeHist(gray),
        clahe.apply(gray),
        cv2.GaussianBlur(gray, (5, 5), 0),
        # sharpen
        cv2.addWeighted(gray, 1.5, cv2.GaussianBlur(gray, (5, 5), 0), -0.5, 0),
        # upscale 2x (helps when chessboard is small in frame)
        cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
    ]
    flags_list = [
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS,
    ]
    for img in attempts:
        for flags in flags_list:
            found, corners = cv2.findChessboardCorners(img, size, flags)
            if found:
                # if upscaled, scale corners back down
                if img.shape[0] == gray.shape[0] * 2:
                    corners = corners / 2.0
                return corners
    return None

print("=== Detection results ===")
best_size = None
best_count = 0

for size in CANDIDATES:
    count = 0
    sample_img = None
    for frame in sample_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = try_detect(gray, size)
        if corners is not None:
            count += 1
            if sample_img is None:
                sample_img = frame.copy()
                cv2.drawChessboardCorners(sample_img, size, corners, True)
    print(f"  {str(size):8s}: {count}/{len(sample_frames)} frames detected")
    if sample_img is not None:
        cv2.imwrite(f"{OUTPUT_DIR}/detected_{size[0]}x{size[1]}.jpg", sample_img)
    if count > best_count:
        best_count = count
        best_size = size

print(f"\nBest: CHESSBOARD_SIZE = {best_size}  ({best_count} detections)")
print(f"Check '{OUTPUT_DIR}/detected_*.jpg' to verify corners visually.")
