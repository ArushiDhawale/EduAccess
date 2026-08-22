import unittest
import os
import json
import tempfile
import numpy as np
from collections import deque

# Import modules to test
from services.hand_landmarks import normalize_landmarks
from services.isl_nlp import text_to_gloss, get_sign_video, load_sign_vocabulary, clear_vocabulary_cache
from services.sign_predictor import PredictionSmoother

class TestISLTranslatorCore(unittest.TestCase):
    
    def setUp(self):
        # Clear vocabulary cache to ensure clean tests
        clear_vocabulary_cache()
        # Set paths relative to workspace
        self.vocab_path = "config/sign_vocabulary.json"
        
    def test_vocabulary_configuration(self):
        """Verify that every vocabulary entry has the correct required fields."""
        self.assertTrue(os.path.exists(self.vocab_path), "Sign vocabulary config file missing.")
        
        vocab = load_sign_vocabulary(self.vocab_path)
        self.assertGreater(len(vocab), 0, "Vocabulary should not be empty.")
        
        required_keys = [
            "HELLO", "PLEASE", "GOOD_MORNING", "NO", "YES",
            "SORRY", "THANK_YOU", "HELP", "WHO", "WHAT"
        ]
        for key in required_keys:
            self.assertIn(key, vocab, f"Required sign {key} not found in vocabulary config.")

        self.assertEqual(set(vocab), set(required_keys))
            
        # Test each entry's schema
        for sign_key, info in vocab.items():
            self.assertIn("display_name", info, f"Missing display_name for {sign_key}")
            self.assertIn("gloss", info, f"Missing gloss for {sign_key}")
            self.assertIn("video", info, f"Missing video for {sign_key}")
            self.assertIsInstance(info.get("aliases", []), list, f"aliases should be a list in {sign_key}")

    def test_landmark_normalization(self):
        """Test that hand landmark normalization returns a consistent 63-element feature vector."""
        # 1. Create a dummy set of 21 landmarks (wrist at origin 0,0,0, middle finger MCP at 0, 10, 0)
        landmarks = np.zeros((21, 3))
        # Set wrist (index 0)
        landmarks[0] = [0.0, 0.0, 0.0]
        # Set middle finger MCP (index 9)
        landmarks[9] = [0.0, 10.0, 0.0]
        # Set other landmarks to some values
        for i in range(1, 21):
            if i != 9:
                landmarks[i] = [float(i), float(i * 2), float(-i)]
                
        # Run normalization
        flat_norm = normalize_landmarks(landmarks)
        
        self.assertIsNotNone(flat_norm)
        self.assertEqual(len(flat_norm), 63, "Normalized vector must contain exactly 63 elements (21 landmarks * 3 coordinates).")
        
        # Verify wrist is at 0,0,0 after normalization
        self.assertEqual(flat_norm[0], 0.0)
        self.assertEqual(flat_norm[1], 0.0)
        self.assertEqual(flat_norm[2], 0.0)
        
        # Verify scale distance: wrist (0) to middle finger MCP (9) should normalize to a length of 1.0
        # Since middle finger MCP is index 9, its normalized coordinates start at 9*3 = 27
        norm_9_x = flat_norm[27]
        norm_9_y = flat_norm[28]
        norm_9_z = flat_norm[29]
        norm_dist = np.sqrt(norm_9_x**2 + norm_9_y**2 + norm_9_z**2)
        self.assertAlmostEqual(norm_dist, 1.0, places=5, msg="Normalized distance should be 1.0")

    def test_text_to_gloss_conversion(self):
        """Test conversion of standard English sentences into corresponding ISL glosses."""
        # Test exact match
        res = text_to_gloss("hello", config_path=self.vocab_path)
        self.assertEqual(res["glosses"], ["HELLO"])
        self.assertEqual(res["unsupported"], [])
        
        # Test punctuation stripping and casing
        res = text_to_gloss("Hello, please!", config_path=self.vocab_path)
        self.assertEqual(res["glosses"], ["HELLO", "PLEASE"])
        self.assertEqual(res["unsupported"], [])
        
        # Test sentence with multiple valid words
        res = text_to_gloss("I need help", config_path=self.vocab_path)
        # "HELP" is supported; "I" and "need" are outside the selected vocabulary.
        self.assertEqual(res["glosses"], ["HELP"])
        self.assertEqual(res["unsupported"], ["i", "need"])

    def test_phrase_matching(self):
        """Verify that multi-word phrases are matched before single words."""
        # "thank you" must become "THANK_YOU", not "THANK" + "YOU" or unsupported
        res = text_to_gloss("thank you", config_path=self.vocab_path)
        self.assertEqual(res["glosses"], ["THANK_YOU"])
        self.assertEqual(res["unsupported"], [])
        
        res = text_to_gloss("good morning", config_path=self.vocab_path)
        self.assertEqual(res["glosses"], ["GOOD_MORNING"])
        self.assertEqual(res["unsupported"], [])
        
        # Complex sentence mixing phrase and word
        res = text_to_gloss("hello thank you please", config_path=self.vocab_path)
        self.assertEqual(res["glosses"], ["HELLO", "THANK_YOU", "PLEASE"])
        self.assertEqual(res["unsupported"], [])

    def test_unsupported_words(self):
        """Verify that unsupported words are explicitly returned in the unsupported list."""
        res = text_to_gloss("hello tomorrow doctor", config_path=self.vocab_path)
        # Only "hello" is in the selected vocabulary.
        self.assertEqual(res["glosses"], ["HELLO"])
        self.assertEqual(res["unsupported"], ["tomorrow", "doctor"])

    def test_prediction_smoothing(self):
        """Verify that unstable predictions are not accepted immediately, but stable ones are."""
        smoother = PredictionSmoother(smoothing_window=8, min_stable_frames=4, confidence_threshold=0.70)
        
        # 1. Low confidence prediction should be ignored
        res1 = smoother.add_prediction({"label": "HELLO", "confidence": 0.50})
        self.assertIsNone(res1, "Low confidence prediction should not produce a stable label.")
        
        # 2. Add unstable predictions (less than min_stable_frames = 4)
        smoother.clear()
        smoother.add_prediction({"label": "HELLO", "confidence": 0.90}) # count=1
        smoother.add_prediction({"label": "HELP", "confidence": 0.85})  # count=1
        res3 = smoother.add_prediction({"label": "HELLO", "confidence": 0.95}) # count=2 (window contains: HELLO, HELP, HELLO)
        self.assertIsNone(res3, "Unstable predictions should not produce a stable label.")
        
        # 3. Add stable predictions (meets min_stable_frames = 4)
        smoother.clear()
        smoother.add_prediction({"label": "HELLO", "confidence": 0.90}) # 1
        smoother.add_prediction({"label": "HELLO", "confidence": 0.95}) # 2
        smoother.add_prediction({"label": "HELLO", "confidence": 0.88}) # 3
        res_stable = smoother.add_prediction({"label": "HELLO", "confidence": 0.92}) # 4
        self.assertEqual(res_stable, "HELLO", "Stable predictions should be accepted once min_stable_frames is met.")

    def test_missing_video_handling(self):
        """Verify that get_sign_video returns None if the video file does not exist, rather than crashing."""
        # Non-existent gloss
        video_path = get_sign_video("NON_EXISTENT_GLOSS", config_path=self.vocab_path)
        self.assertIsNone(video_path, "Non-existent gloss should return None.")
        
        # A configured gloss whose video file is missing on disk
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "vocabulary.json")
            with open(config_path, "w") as config_file:
                json.dump({"MISSING": {"video": os.path.join(temp_dir, "missing.mp4")}}, config_file)
            video_path = get_sign_video("MISSING", config_path=config_path)
            self.assertIsNone(video_path, "Missing video file on disk should return None.")

if __name__ == "__main__":
    unittest.main()
