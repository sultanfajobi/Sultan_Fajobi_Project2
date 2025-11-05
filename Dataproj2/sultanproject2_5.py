# STEP 5 - MODEL TESTING AND EVALUATION (Sultan Fajobi)

import os, json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --------------------------------------------------
# Load class mapping from Steps 1–4
# --------------------------------------------------
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Convert index mapping back to readable list
class_names = list(class_indices.keys())
print("Class order:", class_names)

# --------------------------------------------------
# Detect which trained model to load (V1 or V2)
# --------------------------------------------------
model_path = None
for name in ["sultan_model_v2.keras", "sultan_model_v1.keras", "sultan_model.keras"]:
    if os.path.exists(name):
        model_path = name
        break

if model_path is None:
    raise FileNotFoundError("No trained model found (expected sultan_model_v1.keras or sultan_model_v2.keras).")

print(f"Using model: {model_path}\n")
model = tf.keras.models.load_model(model_path)

# --------------------------------------------------
# Define test images 
# --------------------------------------------------
test_images = {
    "crack": "/Users/sultan/Documents/GitHub/AER850 PROJ1/Sultan_Fajobi_Project2/Dataproj2/Data/Test/crack/test_crack.jpg",
    "missing-head": "/Users/sultan/Documents/GitHub/AER850 PROJ1/Sultan_Fajobi_Project2/Dataproj2/Data/Test/missing-head/test_missinghead.jpg",
    "paint-off": "/Users/sultan/Documents/GitHub/AER850 PROJ1/Sultan_Fajobi_Project2/Dataproj2/Data/Test/paint-off/test_paintoff.jpg"
}

# --------------------------------------------------
# Prediction function
# --------------------------------------------------
def predict_image(img_path):
    img = load_img(img_path, target_size=(500, 500))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    
    preds = model.predict(arr)
    idx = np.argmax(preds)
    conf = preds[0][idx]
    predicted_label = class_names[idx]
    return predicted_label, conf, img

# --------------------------------------------------
# Run predictions and display results
# --------------------------------------------------
for actual_label, path in test_images.items():
    predicted_label, conf, img = predict_image(path)
    
    print(f"[TEST] {os.path.basename(path)}: actual={actual_label}  predicted={predicted_label}  conf={conf:.3f}")
    
    # Show image with label
    plt.figure()
    plt.imshow(img)
    plt.title(f"Actual: {actual_label},  Predicted: {predicted_label} ({conf*100:.1f}%)")
    plt.axis("off")
    plt.show()