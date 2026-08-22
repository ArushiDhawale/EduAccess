import os
import json
import joblib
import numpy as np
from collections import deque, Counter

class ISLSignPredictor:
    def __init__(self, model_path="models/isl_static_classifier.pkl", label_mapping_path="models/label_mapping.json"):
        self.model_path = model_path
        self.label_mapping_path = label_mapping_path
        self.model = None
        self.label_mapping = {}
        self.reverse_mapping = {}
        self.load_model()
        
    def load_model(self):
        """
        Loads the trained machine learning model and label mapping from files.
        """
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
            except Exception as e:
                self.model = None
                print(f"Error loading model: {e}")
        else:
            self.model = None
            
        if os.path.exists(self.label_mapping_path):
            try:
                with open(self.label_mapping_path, 'r') as f:
                    self.label_mapping = json.load(f)
                # Map keys to integers for scikit-learn output indexes
                self.reverse_mapping = {int(k): v for k, v in self.label_mapping.items()}
            except Exception as e:
                self.label_mapping = {}
                self.reverse_mapping = {}
                print(f"Error loading label mapping: {e}")
                
    def is_model_loaded(self):
        return self.model is not None and len(self.reverse_mapping) > 0
        
    def predict(self, normalized_features):
        """
        Predicts the ISL sign from normalized hand features.
        
        Args:
            normalized_features: 63-dimensional feature list/array
        Returns:
            dict containing:
                "label": string representing predicted sign
                "confidence": float probability value
            or None if the model is not loaded/fails
        """
        if not self.is_model_loaded() or normalized_features is None:
            return None
            
        # Reshape for scikit-learn input (1, 63)
        features = np.array(normalized_features).reshape(1, -1)
        
        try:
            # Check if predict_proba is available for confidence estimation
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(features)[0]
                pred_class_idx = np.argmax(probabilities)
                confidence = float(probabilities[pred_class_idx])
                label = self.reverse_mapping.get(pred_class_idx, "UNKNOWN")
            else:
                pred_class_idx = int(self.model.predict(features)[0])
                confidence = 1.0
                label = self.reverse_mapping.get(pred_class_idx, "UNKNOWN")
                
            return {
                "label": label,
                "confidence": confidence
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            return None


class PredictionSmoother:
    def __init__(self, smoothing_window=8, min_stable_frames=4, confidence_threshold=0.70):
        """
        Maintains prediction history and filters out unstable and low-confidence predictions.
        
        Args:
            smoothing_window: maximum size of rolling prediction history
            min_stable_frames: minimum occurrences of the same label in the window to accept it
            confidence_threshold: minimum probability/confidence required to consider a prediction
        """
        self.smoothing_window = smoothing_window
        self.min_stable_frames = min_stable_frames
        self.confidence_threshold = confidence_threshold
        self.history = deque(maxlen=smoothing_window)
        
    def add_prediction(self, prediction_dict):
        """
        Adds a new prediction frame.
        
        Args:
            prediction_dict: dict with 'label' and 'confidence', or None
        Returns:
            str: stable label if detected, or None
        """
        if prediction_dict is None:
            self.history.append(None)
            return None
            
        label = prediction_dict.get("label")
        confidence = prediction_dict.get("confidence", 0.0)
        
        # 1. Filter out low confidence predictions
        if confidence < self.confidence_threshold:
            self.history.append(None)
            return None
            
        self.history.append(label)
        
        # 2. Count frequencies in the rolling window
        non_none_history = [lbl for lbl in self.history if lbl is not None]
        if not non_none_history:
            return None
            
        # Get most common prediction in the window
        counter = Counter(non_none_history)
        most_common_label, count = counter.most_common(1)[0]
        
        # 3. Accept only if it meets the minimum stable frames requirement
        if count >= self.min_stable_frames:
            return most_common_label
            
        return None
        
    def clear(self):
        self.history.clear()
