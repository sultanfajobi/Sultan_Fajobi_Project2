# Step 5 — Model Testing / Inference
# AER850 Section 01 Project 2
# Sultan Fajobi (501106769)

import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- Paths & constants ---
ROOT = Path(__file__).resolve().parent
IMG_SIZE = (500, 500)

# Prefer model;
MODEL_PATHS = [ROOT / "sultan_model.keras"]

model_path = None
for p in MODEL_PATHS:
    if p.exists():
        model_path = p
        break
if model_path is None:
    raise FileNotFoundError("No trained model found")

# Load class mapping saved in Steps 1–4 (ensures correct label order)
class_map_path = ROOT / "class_indices.json"
if class_map_path.exists():
    with open(class_map_path) as f:
        class_to_idx = json.load(f)          # e.g., {'crack':0,'missing-head':1,'paint-off':2}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
else:
    # Fallback (only if mapping file not found)
    print("[WARN] class_indices.json not found; falling back to default class order.")
    class_names = ["crack", "missing-head", "paint-off"]

print("Using model:", model_path.name)
print("Class order:", class_names)

# Define the path to the three test images
TEST = ROOT / "Data" / "Test"
test_images = {
    "crack":       TEST / "crack"        / "test_crack.jpg",
    "missing-head":TEST / "missing-head" / "test_missinghead.jpg",
    "paint-off":   TEST / "paint-off"    / "test_paintoff.jpg",
}

# --- Load the trained model ---
model = tf.keras.models.load_model(str(model_path))

def process_and_predict(image_path: Path):
    """Load an image, preprocess to (500,500)/[0,1], predict class + confidence."""
    if not image_path.exists():
        raise FileNotFoundError(f"Missing test image: {image_path}")
    img = load_img(image_path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)

    preds = model.predict(arr, verbose=0)          # shape (1, num_classes)
    pred_idx = int(np.argmax(preds, axis=1)[0])
    pred_label = class_names[pred_idx]
    confidence = float(preds[0][pred_idx])

    return pred_label, confidence, img

# --- Run predictions & display ---
for actual_label, image_path in test_images.items():
    try:
        pred_label, conf, img = process_and_predict(image_path)
        plt.figure()
        plt.imshow(img)
        plt.title(f"Actual: {actual_label} | Predicted: {pred_label} ({conf*100:.2f}%)")
        plt.axis('off')
        plt.show()
        print(f"[OK] {image_path.name}: actual={actual_label}  predicted={pred_label}  conf={conf:.3f}")
    except FileNotFoundError as e:
        print("[WARN]", e)