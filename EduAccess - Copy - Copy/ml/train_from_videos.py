import csv
import os
import sys

import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.hand_landmarks import HandLandmarkExtractor, normalize_landmarks
from services.isl_nlp import load_sign_vocabulary


def extract_video_dataset(video_dir="assets/signs", output_path="data/isl_dataset.csv", frame_step=3):
    """Extract normalized hand landmarks from labeled MP4 files.

    Each video filename is treated as its label, for example hello.mp4 becomes
    HELLO. Frames without a detected hand are skipped.
    """
    video_paths = sorted(
        os.path.join(video_dir, name)
        for name in os.listdir(video_dir)
        if name.lower().endswith((".mp4", ".mov", ".avi"))
    )
    if not video_paths:
        print(f"No sign videos found in {video_dir}")
        return False

    vocabulary = load_sign_vocabulary()
    video_paths = [
        path for path in video_paths
        if os.path.splitext(os.path.basename(path))[0].upper() in vocabulary
    ]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    headers = ["label"] + [axis + str(index) for index in range(1, 22) for axis in ("x", "y", "z")]
    extractor = HandLandmarkExtractor()
    rows = []
    counts = {}

    try:
        for video_path in video_paths:
            label = os.path.splitext(os.path.basename(video_path))[0].upper()
            capture = cv2.VideoCapture(video_path)
            frame_index = 0
            counts[label] = 0

            while True:
                success, frame = capture.read()
                if not success:
                    break
                if frame_index % frame_step == 0:
                    landmarks, _ = extractor.extract_landmarks(frame)
                    if landmarks is not None:
                        rows.append([label] + normalize_landmarks(landmarks).tolist())
                        counts[label] += 1
                frame_index += 1
            capture.release()

        if not rows:
            print("No hands were detected in the supplied videos.")
            return False

        with open(output_path, "w", newline="") as dataset_file:
            writer = csv.writer(dataset_file)
            writer.writerow(headers)
            writer.writerows(rows)

        print(f"Extracted {len(rows)} landmark samples to {output_path}")
        for label, count in counts.items():
            print(f"  {label}: {count} frames")
        return True
    finally:
        extractor.close()


if __name__ == "__main__":
    extract_video_dataset()