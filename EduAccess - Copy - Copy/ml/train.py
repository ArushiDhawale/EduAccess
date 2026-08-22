import os
import sys
import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Adjust path to import preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.preprocess import preprocess_dataset

def train_model(csv_path="data/isl_dataset.csv", model_dir="models"):
    """
    Loads preprocessed data, encodes labels, trains Random Forest model,
    saves label mapping and trained model to disk.
    """
    print(f"Starting training pipeline. Reading dataset from: {csv_path}")
    
    # 1. Preprocess data
    X, y, report = preprocess_dataset(csv_path)
    if not report["success"]:
        print("Preprocessing failed. Cannot train.")
        for err in report["errors"]:
            print(f"Error: {err}")
        return False
        
    print(f"Preprocessing completed. Valid samples: {report['clean_row_count']}")
    if len(report["warnings"]) > 0:
        for warn in report["warnings"]:
            print(f"Warning: {warn}")
            
    # Check if there are at least two samples to train
    if len(X) < 2:
        print("Error: Dataset must have at least 2 samples to train.")
        return False
        
    # Check if we have multiple classes
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print(f"Warning: Dataset has only one class ({unique_classes[0]}). Minimum 2 classes recommended for realistic classification.")
        
    # 2. Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Create mapping from int classes to sign labels
    label_mapping = {int(i): str(label) for i, label in enumerate(label_encoder.classes_)}
    
    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    # Save label mapping
    mapping_path = os.path.join(model_dir, "label_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(label_mapping, f, indent=2)
    print(f"Saved label mapping to: {mapping_path}")
    
    # 3. Train/Test split
    # If the dataset is tiny, do not use stratify (might cause error if class count is 1)
    stratify = y_encoded if len(unique_classes) > 1 and np.min(np.bincount(y_encoded)) >= 2 else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=stratify
    )
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # 4. Train RandomForest Classifier
    # Using 100 estimators and reasonable depth for light and stable classification
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # 5. Evaluate on train and test set
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Validation/Test Accuracy: {test_acc * 100:.2f}%")
    
    # Save the model
    model_path = os.path.join(model_dir, "isl_static_classifier.pkl")
    joblib.dump(model, model_path)
    print(f"Successfully trained and saved model to: {model_path}")
    
    # Output detailed evaluation report
    print("\n--- Detailed Test Classification Report ---")
    # Convert numeric classes back to text for display
    target_names = [label_mapping[i] for i in sorted(label_mapping.keys()) if i in np.unique(np.concatenate([y_train, y_test]))]
    # Check if target names count matches the unique labels in test set
    unique_in_test = np.unique(y_test)
    labels_to_report = sorted(unique_in_test)
    names_to_report = [label_mapping[i] for i in labels_to_report]
    
    print(classification_report(y_test, y_test_pred, labels=labels_to_report, target_names=names_to_report, zero_division=0))
    
    return True

if __name__ == "__main__":
    csv_arg = "data/isl_dataset.csv"
    if len(sys.argv) > 1:
        csv_arg = sys.argv[1]
    train_model(csv_arg)
