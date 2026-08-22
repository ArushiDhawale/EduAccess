import os
import sys
import pandas as pd
import numpy as np

# Adjust path to import services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.isl_nlp import load_sign_vocabulary

def preprocess_dataset(csv_path="data/isl_dataset.csv", vocabulary_path="config/sign_vocabulary.json"):
    """
    Validates, cleans, and preprocesses the CSV dataset.
    
    Args:
        csv_path: path to the dataset CSV file
        vocabulary_path: path to the vocabulary configuration JSON
    Returns:
        tuple: (X_clean, y_clean, report_dict)
    """
    report = {
        "success": True,
        "errors": [],
        "warnings": [],
        "original_row_count": 0,
        "clean_row_count": 0,
        "label_counts": {}
    }
    
    if not os.path.exists(csv_path):
        report["success"] = False
        report["errors"].append(f"Dataset file not found: {csv_path}")
        return None, None, report
        
    try:
        # Load vocabulary to validate labels
        vocab = load_sign_vocabulary(vocabulary_path)
        valid_labels = set(vocab.keys())
        
        # Load CSV using pandas
        df = pd.read_csv(csv_path)
        report["original_row_count"] = len(df)
        
        if len(df) == 0:
            report["success"] = False
            report["errors"].append("Dataset is empty.")
            return None, None, report
            
        # 1. Validate Columns count (should be 64 columns: label + 63 features)
        expected_cols = 64
        if len(df.columns) != expected_cols:
            report["warnings"].append(f"Expected {expected_cols} columns, but found {len(df.columns)}. Attempting to filter.")
            
        # Ensure 'label' column is present
        if 'label' not in df.columns:
            # Fallback: assume first column is label
            report["warnings"].append("No column named 'label'. Assuming first column is label.")
            df.rename(columns={df.columns[0]: 'label'}, inplace=True)
            
        # Keep only label and the next 63 numeric features
        feature_cols = [c for c in df.columns if c != 'label'][:63]
        if len(feature_cols) < 63:
            report["success"] = False
            report["errors"].append(f"Insufficient feature columns. Found {len(feature_cols)}, expected 63.")
            return None, None, report
            
        # 2. Check for missing values (NaN)
        null_mask = df['label'].isna() | df[feature_cols].isna().any(axis=1)
        null_count = null_mask.sum()
        if null_count > 0:
            report["warnings"].append(f"Found {null_count} rows with missing (NaN) values. Dropping these rows.")
            df = df[~null_mask]
            
        # 3. Validate feature count and numeric types per row
        malformed_mask = []
        for idx, row in df.iterrows():
            is_malformed = False
            # Check if features can be converted to float
            try:
                row_feats = row[feature_cols].values.astype(float)
                if len(row_feats) != 63 or np.isnan(row_feats).any():
                    is_malformed = True
            except (ValueError, TypeError):
                is_malformed = True
            malformed_mask.append(is_malformed)
            
        malformed_mask = np.array(malformed_mask)
        malformed_count = malformed_mask.sum()
        if malformed_count > 0:
            report["warnings"].append(f"Found {malformed_count} rows with malformed numeric data. Dropping these rows.")
            df = df[~malformed_mask]
            
        if len(df) == 0:
            report["success"] = False
            report["errors"].append("All rows were filtered out due to missing or malformed values.")
            return None, None, report
            
        # 4. Validate labels against sign vocabulary
        invalid_labels_mask = ~df['label'].isin(valid_labels)
        invalid_count = invalid_labels_mask.sum()
        if invalid_count > 0:
            unique_invalid = df.loc[invalid_labels_mask, 'label'].unique()
            report["warnings"].append(f"Found {invalid_count} rows with labels not in vocabulary config: {unique_invalid.tolist()}. Keeping them for training but flagging.")
            
        # Reset index after dropping rows
        df.reset_index(drop=True, inplace=True)
        report["clean_row_count"] = len(df)
        
        # Count occurrences per label
        label_counts = df['label'].value_counts().to_dict()
        report["label_counts"] = label_counts
        
        # Split features and labels
        X = df[feature_cols].values.astype(float)
        y = df['label'].values.astype(str)
        
        return X, y, report
        
    except Exception as e:
        report["success"] = False
        report["errors"].append(f"Exception during preprocessing: {e}")
        return None, None, report

if __name__ == "__main__":
    # Test script if executed directly
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "data/isl_dataset.csv"
        
    print(f"Testing preprocessing on {csv_path}...")
    X, y, r = preprocess_dataset(csv_path)
    print("\n--- PREPROCESSING REPORT ---")
    print(f"Success: {r['success']}")
    print(f"Original Rows: {r['original_row_count']}")
    print(f"Cleaned Rows: {r['clean_row_count']}")
    print(f"Errors: {r['errors']}")
    print(f"Warnings: {r['warnings']}")
    print(f"Label Distribution: {r['label_counts']}")
