import argparse
from pathlib import Path

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


def load_artifacts(models_dir: str | Path | None = None, data_dir: str | Path | None = None):
    model_root = Path(models_dir or default_models_dir()).resolve()
    model = tf.keras.models.load_model(model_root / "gesture_model.h5")
    classes = load_class_names(model_root / "classes.txt")
    calorie_map = load_calorie_map(data_dir or default_data_dir())
    return model, classes, calorie_map


def preprocess_image(image_path: str | Path) -> np.ndarray:
    image = tf.keras.utils.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    image_array = tf.keras.utils.img_to_array(image) / 255.0
    return np.expand_dims(image_array, axis=0)


def predict_image(
    image_path: str | Path,
    models_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    top_k: int = 3,
) -> dict[str, object]:
    model, classes, calorie_map = load_artifacts(models_dir=models_dir, data_dir=data_dir)
    probs = model.predict(preprocess_image(image_path), verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:top_k]

    predictions: list[dict[str, object]] = []
    for index in top_indices:
        label = classes[int(index)]
        predictions.append(
            {
                "label": label,
                "display_label": display_label(label),
                "confidence": float(probs[int(index)]),
                "calories": int(estimate_calories(label, calorie_map)),
            }
        )

    return {
        "image_path": str(Path(image_path).resolve()),
        "top_prediction": predictions[0],
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the food label and calories for an image.")
    parser.add_argument("image", help="Path to the image to classify.")
    parser.add_argument("--models-dir", default=str(default_models_dir()), help="Directory containing saved artifacts.")
    parser.add_argument("--data-dir", default=str(default_data_dir()), help="Directory containing calorie_map.json.")
    parser.add_argument("--top-k", type=int, default=3, help="How many predictions to print.")
    args = parser.parse_args()

    result = predict_image(
        image_path=args.image,
        models_dir=args.models_dir,
        data_dir=args.data_dir,
        top_k=args.top_k,
    )

    top = result["top_prediction"]
    print(f"Image: {result['image_path']}")
    print(
        f"Prediction: {top['display_label']} | "
        f"confidence={top['confidence']:.2%} | "
        f"estimated_calories={top['calories']} kcal"
    )
    print("Top candidates:")
    for item in result["predictions"]:
        print(
            f"- {item['display_label']}: "
            f"{item['confidence']:.2%}, "
            f"{item['calories']} kcal"
        )


if __name__ == "__main__":
    main()
