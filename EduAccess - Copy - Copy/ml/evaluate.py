import os
import sys
import json
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Adjust path to import services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.preprocess import preprocess_dataset

def evaluate_model(csv_path="data/isl_dataset.csv", model_path="models/isl_static_classifier.pkl", label_mapping_path="models/label_mapping.json"):
    """
    Evaluates a saved model's performance against a dataset.
    Prints accuracy, precision, recall, F1-score, and confusion matrix.
    
    Returns:
        dict containing metrics or None if evaluation failed.
    """
    print(f"Starting evaluation of model: {model_path}")
    print(f"Evaluating against dataset: {csv_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
        
    if not os.path.exists(label_mapping_path):
        print(f"Error: Label mapping not found at {label_mapping_path}")
        return None
        
    # Load model and mapping
    try:
        model = joblib.load(model_path)
        with open(label_mapping_path, "r") as f:
            label_mapping = json.load(f)
        reverse_mapping = {int(k): v for k, v in label_mapping.items()}
    except Exception as e:
        print(f"Error loading model assets: {e}")
        return None
        
    # Load and preprocess dataset
    X, y, preprocess_report = preprocess_dataset(csv_path)
    if not preprocess_report["success"]:
        print(f"Error: Preprocessing failed: {preprocess_report['errors']}")
        return None
        
    # Translate text labels to encoded integers as expected by the model
    # Convert label mapping back to label-to-index for encoding
    label_to_index = {v: k for k, v in reverse_mapping.items()}
    
    y_encoded = []
    skipped_rows = 0
    X_valid = []
    
    for i, label in enumerate(y):
        if label in label_to_index:
            y_encoded.append(label_to_index[label])
            X_valid.append(X[i])
        else:
            skipped_rows += 1
            
    if skipped_rows > 0:
        print(f"Warning: Skipped {skipped_rows} samples because their labels were not in the model's label mapping.")
        
    if len(y_encoded) == 0:
        print("Error: No valid test samples matches the model's known classes.")
        return None
        
    X_valid = np.array(X_valid)
    y_encoded = np.array(y_encoded)
    
    # Run predictions
    try:
        y_pred = model.predict(X_valid)
    except Exception as e:
        print(f"Prediction failed during evaluation: {e}")
        return None
        
    # Calculate metrics
    accuracy = accuracy_score(y_encoded, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_encoded, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_encoded, y_pred)
    
    # Generate classification report
    labels_present = sorted(list(set(y_encoded.tolist()) | set(y_pred.tolist())))
    target_names = [reverse_mapping[i] for i in labels_present]
    
    cls_report = classification_report(
        y_encoded, y_pred,
        labels=labels_present,
        target_names=target_names,
        zero_division=0
    )
    
    print("\n================ EVALUATION METRICS ================")
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1 Score : {f1 * 100:.2f}%")
    print("====================================================")
    print("\n--- Confusion Matrix ---")
    print(conf_matrix)
    print("\n--- Classification Report ---")
    print(cls_report)
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": conf_matrix.tolist()
    }

if __name__ == "__main__":
    csv_arg = "data/isl_dataset.csv"
    if len(sys.argv) > 1:
        csv_arg = sys.argv[1]
    evaluate_model(csv_arg)
