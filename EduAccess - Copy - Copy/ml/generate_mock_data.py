import os
import sys
import csv
import math
import random
import numpy as np

# Adjust path to import services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.isl_nlp import load_sign_vocabulary

def generate_synthetic_dataset(output_path="data/isl_dataset.csv", vocabulary_path="config/sign_vocabulary.json", samples_per_sign=30):
    """
    Generates a synthetic hand landmark dataset for demo/testing purposes.
    This creates distinctive landmark configurations for each sign in the vocabulary.
    
    Args:
        output_path: file path to save the generated CSV
        vocabulary_path: path to the vocabulary configuration JSON
        samples_per_sign: number of samples to generate per sign
    Returns:
        bool: True if successful, False otherwise
    """
    vocab = load_sign_vocabulary(vocabulary_path)
    if not vocab:
        print("Error: Vocabulary configuration could not be loaded.")
        return False
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    headers = ["label"]
    for i in range(1, 22):
        headers.extend([f"x{i}", f"y{i}", f"z{i}"])
        
    try:
        with open(output_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for gloss in vocab.keys():
                # Unique seed base for each sign to make poses distinct
                seed_base = sum(ord(c) for c in gloss)
                
                for sample_idx in range(samples_per_sign):
                    # Add sample index to seed to get variation across samples
                    np.random.seed(seed_base + sample_idx)
                    random.seed(seed_base + sample_idx)
                    
                    base_pose = []
                    # Generate 21 landmarks
                    for i in range(21):
                        # Construct a unique shape signature per sign gloss using math curves
                        # This generates reproducible, distinct shapes per sign
                        bx = math.sin(i * 0.4 + seed_base * 0.15) * 0.6
                        by = math.cos(i * 0.4 - seed_base * 0.1) * 0.8
                        bz = math.sin(i * 0.2 + seed_base * 0.3) * 0.3
                        base_pose.append([bx, by, bz])
                        
                    base_pose = np.array(base_pose)
                    
                    # 1. Translate wrist (landmark 0) to origin
                    base_pose = base_pose - base_pose[0]
                    
                    # 2. Scale landmarks based on wrist (0) to middle MCP (9) distance
                    scale_dist = np.linalg.norm(base_pose[9])
                    if scale_dist > 1e-6:
                        base_pose = base_pose / scale_dist
                        
                    # 3. Add small random gaussian noise to simulate hand shaking
                    noise = np.random.normal(0, 0.04, base_pose.shape)
                    # Wrist has zero noise so it stays at the origin
                    noise[0] = [0.0, 0.0, 0.0]
                    
                    pose = base_pose + noise
                    
                    # 4. Re-normalize to ensure strict adherence to normalization rules
                    pose = pose - pose[0]
                    final_scale = np.linalg.norm(pose[9])
                    if final_scale > 1e-6:
                        pose = pose / final_scale
                        
                    # Write row: label followed by 63 features
                    row = [gloss] + list(pose.flatten())
                    writer.writerow(row)
                    
        print(f"Synthetic dataset with {len(vocab) * samples_per_sign} samples successfully saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error generating synthetic dataset: {e}")
        return False

if __name__ == "__main__":
    generate_synthetic_dataset()
