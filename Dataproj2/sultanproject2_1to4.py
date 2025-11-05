#!# Steps 1 to 4
# AER850 Section 01 Project 2
# Sultan Fajobi (501106769)

'''STEP 1 - DATA PROCESSING'''
import os, json, random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Define image size and batch size
IMG_SIZE = (500, 500)
BATCH_SIZE = 32

# Define paths
train_dir = '/Users/sultan/Documents/GitHub/AER850 PROJ1/Sultan_Fajobi_Project2/Dataproj2/Data/Train'
val_dir   = '/Users/sultan/Documents/GitHub/AER850 PROJ1/Sultan_Fajobi_Project2/Dataproj2/Data/Valid'
test_dir  = '/Users/sultan/Documents/GitHub/AER850 PROJ1/Sultan_Fajobi_Project2/Dataproj2/Data/Test'  # not used yet in Steps 1–4

# Data Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2
)
val_datagen = ImageDataGenerator(rescale=1.0/255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=SEED
)
validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Prints for report
print("Class indices:", train_generator.class_indices)
print("Train samples:", train_generator.samples, " | Val samples:", validation_generator.samples)

# Save class mapping for Step 5
with open("class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f, indent=2)

'''STEP 2 & 3 - NEURAL NETWORK ARCHITECTURE DESIGN and HYPERPARAMETER ANALYSIS'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout

# --- Model V1---
def build_model_v1():
    model = Sequential([
        Input(shape=(500, 500, 3)),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),

        Dense(3, activation='softmax')  # 3 classes: crack, missing-head, paint-off
    ])
    return model

# --- Model V2: Minimal variation (BatchNorm after convs + small Dropout in dense) ---
def build_model_v2():
    model = Sequential([
        Input(shape=(500, 500, 3)),
        Conv2D(16, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),

        Dense(3, activation='softmax')
    ])
    return model

import pandas as pd
import matplotlib.pyplot as plt

def compile_train_save(model, suffix):
    # Compile (same hyperparameters for both variants)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Summary 
    print(f"\n===== MODEL SUMMARY ({suffix}) =====")
    model.summary()

    # Train full 25 epochs 
    history = model.fit(
        train_generator,
        epochs=25,
        validation_data=validation_generator,
        verbose=1
    )

    # Save model & training history
    model.save(f'sultan_model_{suffix}.keras')
    pd.DataFrame(history.history).to_csv(f'sultan_history_{suffix}.csv', index=False)

    # Plot training curves for this variant
    plt.figure(figsize=(12, 4))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.title(f'Model Accuracy ({suffix.upper()})'); plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title(f'Model Loss ({suffix.upper()})'); plt.legend()

    plt.tight_layout()
    plt.savefig(f'sultan_training_curves_{suffix}.png', dpi=300)  # high-res for report
    plt.show()

    return history

# ----- Train both variants -----
model_v1 = build_model_v1()
hist_v1  = compile_train_save(model_v1, 'v1')

model_v2 = build_model_v2()
hist_v2  = compile_train_save(model_v2, 'v2')

# combined comparison plot (overlay on one figure)
plt.figure(figsize=(12,4))
# Accuracy overlay
plt.subplot(1,2,1)
plt.plot(hist_v1.history['val_accuracy'], label='V1 Val Acc')
plt.plot(hist_v2.history['val_accuracy'], label='V2 Val Acc')
plt.xlabel('Epochs'); plt.ylabel('Val Accuracy'); plt.title('Validation Accuracy: V1 vs V2'); plt.legend()

# Loss overlay
plt.subplot(1,2,2)
plt.plot(hist_v1.history['val_loss'], label='V1 Val Loss')
plt.plot(hist_v2.history['val_loss'], label='V2 Val Loss')
plt.xlabel('Epochs'); plt.ylabel('Val Loss'); plt.title('Validation Loss: V1 vs V2'); plt.legend()

plt.tight_layout()
plt.savefig('sultan_training_curves_compare.png', dpi=300)
plt.show()