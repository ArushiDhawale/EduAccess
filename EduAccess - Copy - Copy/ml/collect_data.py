import os
import sys
import time
import argparse
import csv
import cv2
import numpy as np

# Adjust path to import services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.hand_landmarks import HandLandmarkExtractor, normalize_landmarks

def main():
    parser = argparse.ArgumentParser(description="Collect hand landmarks dataset from camera.")
    parser.add_argument("--label", type=str, required=True, help="The ISL sign label to collect data for (e.g., HELLO).")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to collect.")
    parser.add_argument("--output", type=str, default="data/isl_dataset.csv", help="Path to output CSV file.")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between sample collections in seconds.")
    args = parser.parse_args()

    # Upper-case label
    label = args.label.upper()
    
    # Create data directory if not exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize extractor
    extractor = HandLandmarkExtractor(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    
    # Initialize CSV file with headers if it does not exist
    file_exists = os.path.exists(args.output)
    
    headers = ["label"]
    for i in range(1, 22):
        headers.extend([f"x{i}", f"y{i}", f"z{i}"])
        
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
        
    print(f"--- Data Collection for sign: {label} ---")
    print("Instructions:")
    print("  1. Position your hand in front of the camera showing the sign.")
    print("  2. Press 's' to START automatic collection.")
    print("  3. Press 'q' to quit.")
    print("---------------------------------------")
    
    started = False
    samples_collected = 0
    
    while samples_collected < args.num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        # Flip frame horizontally for natural mirror view
        frame = cv2.flip(frame, 1)
        
        # Extract landmarks
        landmarks, annotated_frame = extractor.extract_landmarks(frame)
        
        # UI overlays
        status_text = f"Samples: {samples_collected}/{args.num_samples}"
        if not started:
            cv2.putText(annotated_frame, "Press 's' to Start", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(annotated_frame, "COLLECTING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        cv2.putText(annotated_frame, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.imshow("Data Collection", annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Collection cancelled by user.")
            break
        elif key == ord('s'):
            started = True
            
        if started and landmarks is not None:
            # We normalize landmarks
            try:
                flat_norm = normalize_landmarks(landmarks)
                # Write to CSV
                with open(args.output, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    if not file_exists and os.path.getsize(args.output) == 0:
                        writer.writerow(headers)
                    row = [label] + list(flat_norm)
                    writer.writerow(row)
                samples_collected += 1
                time.sleep(args.delay)
            except Exception as e:
                print(f"Error saving sample: {e}")
                
    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print(f"Done! Collected {samples_collected} samples for {label} and saved to {args.output}")

if __name__ == "__main__":
    main()
