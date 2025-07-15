import sys
import os
# Add project root to path to find other modules like utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import glob
import shutil
import hashlib
import numpy as np
import onnxruntime as ort
from PIL import Image
from neo4j import GraphDatabase
from utils.id_helpers import get_iso_utc_now # Re-using our time helper

# --- CONFIGURATION ---
CONFIG = {
    "inbox_path": "data/images/inbox",
    "processed_path": "data/images/processed", # Create this folder if it doesn't exist
    "onnx_model_path": "vision/mosvit_blite_int8.onnx",
    "image_size": 224,
    "class_map": {0: 'healthy', 1: 'lateblt'},
    "field_id": "FIELD_42" # Hard-coded for this prototype
}

def preprocess_image(img_path):
    """Loads and preprocesses a single image for ONNX inference."""
    img = Image.open(img_path).convert('RGB').resize((CONFIG["image_size"], CONFIG["image_size"]))
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_np = (img_np - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img_np = np.transpose(img_np, (2, 0, 1))
    img_np = np.expand_dims(img_np, axis=0).astype(np.float32)
    return img_np

def softmax(x):
    """Compute softmax values for a set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)

def run_ingestion():
    print("Starting ingestion script...")
    # Ensure the processed directory exists
    if not os.path.exists(CONFIG["processed_path"]):
        os.makedirs(CONFIG["processed_path"])

    # Load the ONNX model
    ort_session = ort.InferenceSession(CONFIG["onnx_model_path"])
    input_name = ort_session.get_inputs()[0].name

    # Find images to process
    images_to_process = glob.glob(os.path.join(CONFIG["inbox_path"], '*.jpg'))
    print(f"Found {len(images_to_process)} images to process.")

    if not images_to_process:
        return

    # Connect to Neo4j
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j","test1234"))
    with driver.session(database="neo4j") as sess:
        query = """
        MERGE (f:Field {id: $field_id})
        MERGE (ls:LeafStatus {id: $leaf_status_id})
            ON CREATE SET
                ls.label = $label,
                ls.probability = $probability,
                ls.observed_at = datetime($observed_at),
                ls.source_image = $image_name
        MERGE (f)-[:HAD_STATUS]->(ls)
        """
        for img_path in images_to_process:
            image_filename = os.path.basename(img_path)
            print(f"  - Processing {image_filename}...")

            # 1. Preprocess image and run inference
            processed_img = preprocess_image(img_path)
            ort_outs = ort_session.run(None, {input_name: processed_img})

            # 2. Post-process results
            probabilities = softmax(ort_outs[0])[0]
            predicted_class_id = np.argmax(probabilities)
            predicted_label = CONFIG["class_map"][predicted_class_id]
            confidence = float(probabilities[predicted_class_id])

            # 3. Build data and write to Neo4j
            leaf_status_id = hashlib.sha1(img_path.encode()).hexdigest()
            observed_at_ts = get_iso_utc_now()

            sess.run(query, {
                "field_id": CONFIG["field_id"],
                "leaf_status_id": leaf_status_id,
                "label": predicted_label,
                "probability": confidence,
                "observed_at": observed_at_ts,
                "image_name": image_filename
            })
            print(f"    -> Classified as '{predicted_label}' with {confidence:.2f} confidence. Wrote to Neo4j.")

            # 4. Move processed file
            shutil.move(img_path, os.path.join(CONFIG["processed_path"], image_filename))

    driver.close()
    print("\nIngestion complete.")

if __name__ == "__main__":
    run_ingestion()