import sys
import os
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import pandas as pd
import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
CONFIG = {
    "labels_csv_path": "data/labels.csv",
    "onnx_model_path": "vision/mosvit_blite_int8.onnx",
    "image_size": 224,
    "class_map": {0: 'healthy', 1: 'lateblt'},
    "report_path": "vision/validation_report.md",
    "confusion_matrix_path": "vision/confusion_matrix.png"
}

def preprocess_image(img_path):
    """Loads and preprocesses a single image for ONNX inference."""
    img = Image.open(img_path).convert('RGB').resize((CONFIG["image_size"], CONFIG["image_size"]))
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_np = (img_np - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img_np = np.transpose(img_np, (2, 0, 1))
    return np.expand_dims(img_np, axis=0).astype(np.float32)

def softmax(x):
    """Compute softmax values for a set of scores."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)

def run_validation():
    print("Starting validation...")

    ort_session = ort.InferenceSession(CONFIG["onnx_model_path"])
    input_name = ort_session.get_inputs()[0].name

    full_df = pd.read_csv(CONFIG["labels_csv_path"])

    # --- THIS IS THE FIX: Normalize path separators before filtering ---
    test_df = full_df[full_df['image_path'].str.replace('\\', '/').str.contains('/test/')].copy()

    print(f"Found {len(test_df)} images in the test set to validate.")

    true_labels = []
    predicted_labels = []
    misclassified = []

    for index, row in test_df.iterrows():
        img_path = row['image_path']
        true_label_str = row['class_label']

        try:
            processed_img = preprocess_image(img_path)
            ort_outs = ort_session.run(None, {input_name: processed_img})
            probabilities = softmax(ort_outs[0])[0]
            predicted_class_id = np.argmax(probabilities)
            predicted_label_str = CONFIG["class_map"][predicted_class_id]

            true_labels.append(true_label_str)
            predicted_labels.append(predicted_label_str)

            if true_label_str != predicted_label_str:
                misclassified.append({
                    "image": img_path,
                    "true_label": true_label_str,
                    "predicted_label": predicted_label_str,
                    "confidence": float(probabilities[predicted_class_id])
                })
        except Exception as e:
            print(f"Could not process {img_path}: {e}")

    if not true_labels:
        print("No valid test images were processed. Exiting.")
        return

    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='macro')

    report_content = f"# Validation Report\n\n"
    report_content += f"## Performance Metrics\n"
    report_content += f"- **Accuracy:** {accuracy:.4f}\n"
    report_content += f"- **F1 Score (Macro):** {f1:.4f}\n\n"

    print("\n--- Validation Complete ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Macro): {f1:.4f}")

    if misclassified:
        print(f"\nFound {len(misclassified)} misclassified images.")
        report_content += f"## Misclassified Images\n"
        report_content += "| Image Path | True Label | Predicted Label | Confidence |\n"
        report_content += "|---|---|---|---|\n"
        for item in misclassified[:20]:
            report_content += f"| {item['image']} | {item['true_label']} | {item['predicted_label']} | {item['confidence']:.2f} |\n"

    with open(CONFIG["report_path"], "w") as f:
        f.write(report_content)
    print(f"\nValidation report saved to {CONFIG['report_path']}")

    cm = confusion_matrix(true_labels, predicted_labels, labels=['healthy', 'lateblt'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['healthy', 'lateblt'], yticklabels=['healthy', 'lateblt'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(CONFIG["confusion_matrix_path"])
    print(f"Confusion matrix saved to {CONFIG['confusion_matrix_path']}")

if __name__ == "__main__":
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nSeaborn and Matplotlib not found.")
        print("Please run: pip install seaborn matplotlib")
        sys.exit(1)

    run_validation()