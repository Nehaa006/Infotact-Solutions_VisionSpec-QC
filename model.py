import tensorflow as tf
from tensorflow.keras import layers, models
from week1_preprocessing import get_data_generators, DATASET_DIR

def build_90_plus_model():
    # Load base
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights='imagenet'
    )
    
    # PHASE 1: UNFREEZE MORE BRAIN POWER
    base_model.trainable = True
    # Freeze only the first 50 layers, let the other 100+ learn PCB features
    for layer in base_model.layers[:50]:
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(), # Added for stability at high accuracy
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4), # Prevent memorizing the boards
        layers.Dense(6, activation='softmax')
    ])

    # PHASE 2: BETTER OPTIMIZATION
    # Start with a slightly higher rate to get out of that 33% rut
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    train_gen, val_gen = get_data_generators(DATASET_DIR)
    model = build_90_plus_model()

    # PHASE 3: LONG-TERM TRAINING
    # Added EarlyStopping so it stops exactly when it hits peak accuracy
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
    ]

    print("🚀 Launching High-Accuracy Training...")
    history = model.fit(
        train_gen, 
        validation_data=val_gen, 
        epochs=50, # Give it time to actually reach 90%
        callbacks=callbacks
    )