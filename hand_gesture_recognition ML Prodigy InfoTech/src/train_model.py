import argparse
import json
from pathlib import Path

import tensorflow as tf

from src.calorie_lookup import build_calorie_map
from src.preprocess import (
    IMG_SIZE,
    build_datasets,
    default_data_dir,
    default_models_dir,
    ensure_project_structure,
    save_calorie_map,
    save_class_names,
)


def build_model(num_classes: int) -> tf.keras.Model:
    augment = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.15),
        ],
        name="augmentation",
    )

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augment(inputs)

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,
        input_tensor=x,
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="food_class")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="food_calorie_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    seed: int = 42,
) -> dict[str, Path | float]:
    ensure_project_structure()
    data_root = Path(data_dir or default_data_dir()).resolve()
    models_root = Path(output_dir or default_models_dir()).resolve()
    artifacts_dir = models_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, class_names, layout = build_datasets(
        data_dir=data_root,
        batch_size=batch_size,
        seed=seed,
    )

    calorie_map = build_calorie_map(class_names)
    calorie_map_path = save_calorie_map(calorie_map, data_root)
    classes_path = save_class_names(class_names, artifacts_dir / "food_classes.txt")
    model_path = artifacts_dir / "food_model.keras"
    metrics_path = artifacts_dir / "metrics.json"

    model = build_model(num_classes=len(class_names))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )

    loss, accuracy = model.evaluate(val_ds, verbose=0)
    model.save(model_path)

    payload = {
        "dataset_layout": str(layout["type"]),
        "data_dir": str(data_root),
        "model_path": str(model_path),
        "classes_path": str(classes_path),
        "calorie_map_path": str(calorie_map_path),
        "val_loss": float(loss),
        "val_accuracy": float(accuracy),
        "epochs_ran": len(history.history.get("loss", [])),
    }

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return {
        "model_path": model_path,
        "classes_path": classes_path,
        "calorie_map_path": calorie_map_path,
        "metrics_path": metrics_path,
        "val_accuracy": float(accuracy),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a food recognition model with calorie estimates.")
    parser.add_argument("--data-dir", default=str(default_data_dir()), help="Dataset root directory.")
    parser.add_argument("--output-dir", default=str(default_models_dir()), help="Model output directory.")
    parser.add_argument("--epochs", type=int, default=10, help="Maximum number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    result = train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print(f"Model saved to: {result['model_path']}")
    print(f"Class labels saved to: {result['classes_path']}")
    print(f"Calorie map saved to: {result['calorie_map_path']}")
    print(f"Validation accuracy: {result['val_accuracy']:.4f}")


if __name__ == "__main__":
    main()
