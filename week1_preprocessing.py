import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Configuration constants for MobileNetV2 [cite: 115]
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
# POINT TO THE NEW STRUCTURED FOLDER
DATASET_DIR = 'pcb-defect-dataset/train_structured' 

def get_data_generators(base_path):
    """
    Implements the full Week 1 Image Pipeline.
    """
    # 1. Define Augmentation and MANDATORY Normalization (0-1)
    train_datagen = ImageDataGenerator(
        rescale=1./255,            # Normalization (0-1) 
        rotation_range=20,         # Real-time Augmentation [cite: 116]
        zoom_range=0.15,           # Real-time Augmentation [cite: 116]
        brightness_range=[0.8, 1.2],# Real-time Augmentation [cite: 116]
        horizontal_flip=True,
        validation_split=0.2       # Prepares for Week 2 validation 
    )

    # 2. Flow from Directory (Ingestion & Standardization)
    train_generator = train_datagen.flow_from_directory(
        base_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH), # Standardized resizing 
        batch_size=BATCH_SIZE,
        class_mode='categorical',             # For multi-class detection
        subset='training',
        shuffle=True
    )

    val_generator = train_datagen.flow_from_directory(
        base_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, val_generator

if __name__ == "__main__":
    if os.path.exists(DATASET_DIR):
        train_gen, val_gen = get_data_generators(DATASET_DIR)
        print(f"\nSUCCESS: Found {train_gen.num_classes} classes.")
        print(f"Class Mapping: {train_gen.class_indices}")
    else:
        print(f"ERROR: Path '{DATASET_DIR}' not found. Run reorganize_data.py first.")