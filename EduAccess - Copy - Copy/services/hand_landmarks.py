import cv2
import mediapipe as mp
import numpy as np
import os

class HandLandmarkExtractor:
    def __init__(self, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "assets",
            "hand_landmarker.task",
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MediaPipe hand model not found at {model_path}. "
                "Download hand_landmarker.task into assets."
            )

        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.hands = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def extract_landmarks(self, frame):
        """
        Extracts raw landmarks for the first detected hand.
        Args:
            frame: opencv BGR image (numpy array)
        Returns:
            landmarks: shape (21, 3) or None if no hand detected
            annotated_frame: copy of frame with landmarks drawn on it
        """
        if frame is None:
            return None, None
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.hands.detect(mp_image)
        
        landmarks = None
        annotated_frame = frame.copy()
        
        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])

            height, width = annotated_frame.shape[:2]
            for landmark in hand_landmarks:
                point = (int(landmark.x * width), int(landmark.y * height))
                cv2.circle(annotated_frame, point, 4, (0, 255, 0), -1)
            
        return landmarks, annotated_frame

    def close(self):
        self.hands.close()


def normalize_landmarks(landmarks):
    """
    Normalizes 21 hand landmarks (shape: 21x3 or list of 21 items with x,y,z).
    1. Wrist (index 0) is the origin.
    2. Subtract wrist coordinates from all landmarks.
    3. Normalize by the distance between wrist (0) and middle finger MCP (9).
    4. Flatten to a 63-dimensional feature vector.
    
    Args:
        landmarks: numpy array of shape (21, 3) or similar
    Returns:
        flat_normalized: 1D numpy array of 63 elements, or None
    """
    if landmarks is None:
        return None
    
    # Convert to numpy array
    coords = np.array(landmarks)
    if coords.shape != (21, 3):
        raise ValueError(f"Landmarks must have shape (21, 3), got {coords.shape}")
        
    wrist = coords[0]
    # Subtract wrist coordinates to translate the origin to the wrist
    shifted = coords - wrist
    
    # Calculate hand scale distance: wrist (0) to middle finger MCP (9)
    scale_dist = np.linalg.norm(shifted[9])
    
    # Avoid division by zero
    if scale_dist < 1e-6:
        scale_dist = 1.0
        
    normalized = shifted / scale_dist
    return normalized.flatten()
