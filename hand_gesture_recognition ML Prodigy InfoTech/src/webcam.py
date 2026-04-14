import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from src.calorie_lookup import estimate_calories
from src.preprocess import (
    IMG_SIZE,
    default_data_dir,
    default_models_dir,
    display_label,
    load_calorie_map,
    load_class_names,
)


def load_runtime(models_dir: str | Path | None = None, data_dir: str | Path | None = None):
    model_root = Path(models_dir or default_models_dir()).resolve() / "artifacts"
    model = tf.keras.models.load_model(model_root / "food_model.keras")
    classes = load_class_names(model_root / "food_classes.txt")
    calorie_map = load_calorie_map(data_dir or default_data_dir())
    return model, classes, calorie_map


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)


def run_webcam(
    camera_index: int = 0,
    models_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> None:
    model, classes, calorie_map = load_runtime(models_dir=models_dir, data_dir=data_dir)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam at camera index {camera_index}. "
            "Check that a camera is connected, available, or try another index."
        )

    print("Press ESC to exit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam.")

            probabilities = model.predict(preprocess_frame(frame), verbose=0)[0]
            index = int(np.argmax(probabilities))
            label = classes[index]
            confidence = float(probabilities[index])
            calories = estimate_calories(label, calorie_map)

            cv2.putText(
                frame,
                f"Food: {display_label(label)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Confidence: {confidence:.1%}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Estimated Calories: {calories} kcal",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            cv2.imshow("Food Recognition & Calorie Estimation", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live food recognition from the webcam.")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=int(os.environ.get("CAMERA_INDEX", "0")),
        help="Camera index to open, for example 0 or 1.",
    )
    parser.add_argument("--models-dir", default=str(default_models_dir()), help="Directory containing saved artifacts.")
    parser.add_argument("--data-dir", default=str(default_data_dir()), help="Directory containing calorie_map.json.")
    args = parser.parse_args()

    run_webcam(camera_index=args.camera_index, models_dir=args.models_dir, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
