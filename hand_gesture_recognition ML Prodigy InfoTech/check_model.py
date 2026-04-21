import tensorflow as tf
from pathlib import Path

model_path = Path("models/gesture_model.h5")
if model_path.exists():
    model = tf.keras.models.load_model(model_path)
    model.summary()
    print("\nInput shape:", model.input_shape)
else:
    print("Model not found")
