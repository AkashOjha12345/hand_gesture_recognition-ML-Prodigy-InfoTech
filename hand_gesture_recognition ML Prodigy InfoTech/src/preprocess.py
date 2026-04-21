import json
from pathlib import Path

import tensorflow as tf

IMG_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE
DEFAULT_CALORIES = 320


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_dir() -> Path:
    return project_root() / "data"


def default_models_dir() -> Path:
    return project_root() / "models"


def ensure_project_structure() -> dict[str, Path]:
    root = project_root()
    data_dir = default_data_dir()
    models_dir = default_models_dir()
    artifacts_dir = models_dir / "artifacts"

    for directory in (data_dir, models_dir, artifacts_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "root": root,
        "data": data_dir,
        "models": models_dir,
        "artifacts": artifacts_dir,
    }


def normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_").replace("-", "_")


def display_label(label: str) -> str:
    return label.replace("_", " ").title()


def dataset_layout(data_dir: str | Path | None = None) -> dict[str, Path | str]:
    resolved = Path(data_dir or default_data_dir()).resolve()
    kaggle_root = resolved / "food-101"
    meta_dir = kaggle_root / "meta"
    images_dir = kaggle_root / "images"
    train_meta = meta_dir / "train.txt"
    test_meta = meta_dir / "test.txt"

    if images_dir.exists() and train_meta.exists():
        return {
            "type": "metadata",
            "root": kaggle_root,
            "images_dir": images_dir,
            "train_meta": train_meta,
            "test_meta": test_meta,
        }

    train_dir = resolved / "train"
    test_dir = resolved / "test"

    if train_dir.exists() and any(path.is_dir() for path in train_dir.iterdir()):
        return {
            "type": "folder",
            "root": resolved,
            "train_dir": train_dir,
            "test_dir": test_dir,
        }

    raise FileNotFoundError(
        "Could not find a supported dataset layout. Expected either "
        "'data/train/<class_name>/*.jpg' or "
        "'data/food-101/{images,meta/train.txt,meta/test.txt}'."
    )


def class_names_from_layout(layout: dict[str, Path | str]) -> list[str]:
    if layout["type"] == "folder":
        train_dir = Path(layout["train_dir"])
        return sorted(path.name for path in train_dir.iterdir() if path.is_dir())

    images_dir = Path(layout["images_dir"])
    return sorted(path.name for path in images_dir.iterdir() if path.is_dir())


def _decode_image(image_path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def _folder_datasets(
    layout: dict[str, Path | str],
    batch_size: int,
    image_size: tuple[int, int],
    seed: int,
    validation_split: float,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    train_dir = Path(layout["train_dir"])
    test_dir = Path(layout["test_dir"])

    class_names = sorted(path.name for path in train_dir.iterdir() if path.is_dir())

    if test_dir.exists() and any(path.is_dir() for path in test_dir.iterdir()):
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            labels="inferred",
            label_mode="int",
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            labels="inferred",
            label_mode="int",
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False,
            class_names=class_names,
        )
        return train_ds, val_ds, class_names

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset="training",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset="validation",
    )
    return train_ds, val_ds, train_ds.class_names


def _metadata_split_to_examples(
    split_file: Path,
    images_dir: Path,
    class_to_index: dict[str, int],
) -> tuple[list[str], list[int]]:
    image_paths: list[str] = []
    labels: list[int] = []

    with split_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            relative = line.strip()
            if not relative:
                continue

            class_name = normalize_label(relative.split("/")[0])
            image_path = images_dir / f"{relative}.jpg"
            if image_path.exists():
                image_paths.append(str(image_path))
                labels.append(class_to_index[class_name])

    return image_paths, labels


def _metadata_dataset(
    image_paths: list[str],
    labels: list[int],
    batch_size: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(image_paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_decode_image, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def _metadata_datasets(
    layout: dict[str, Path | str],
    batch_size: int,
    seed: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    images_dir = Path(layout["images_dir"])
    train_meta = Path(layout["train_meta"])
    test_meta = Path(layout["test_meta"])
    class_names = sorted(path.name for path in images_dir.iterdir() if path.is_dir())
    class_to_index = {name: idx for idx, name in enumerate(class_names)}

    train_paths, train_labels = _metadata_split_to_examples(train_meta, images_dir, class_to_index)
    if not train_paths:
        raise RuntimeError(f"No training samples found in {train_meta}")

    if test_meta.exists():
        val_paths, val_labels = _metadata_split_to_examples(test_meta, images_dir, class_to_index)
    else:
        split_at = max(1, int(len(train_paths) * 0.9))
        val_paths = train_paths[split_at:]
        val_labels = train_labels[split_at:]
        train_paths = train_paths[:split_at]
        train_labels = train_labels[:split_at]

    train_ds = _metadata_dataset(train_paths, train_labels, batch_size, training=True, seed=seed)
    val_ds = _metadata_dataset(val_paths, val_labels, batch_size, training=False, seed=seed)
    return train_ds, val_ds, class_names


def build_datasets(
    data_dir: str | Path | None = None,
    batch_size: int = 32,
    seed: int = 42,
    validation_split: float = 0.15,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str], dict[str, Path | str]]:
    layout = dataset_layout(data_dir)
    image_size = (IMG_SIZE, IMG_SIZE)

    if layout["type"] == "folder":
        train_ds, val_ds, class_names = _folder_datasets(
            layout=layout,
            batch_size=batch_size,
            image_size=image_size,
            seed=seed,
            validation_split=validation_split,
        )
    else:
        train_ds, val_ds, class_names = _metadata_datasets(
            layout=layout,
            batch_size=batch_size,
            seed=seed,
        )

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names, layout


def calorie_map_path(data_dir: str | Path | None = None) -> Path:
    return Path(data_dir or default_data_dir()).resolve() / "calorie_map.json"


def load_calorie_map(data_dir: str | Path | None = None) -> dict[str, int]:
    path = calorie_map_path(data_dir)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return {normalize_label(key): int(value) for key, value in payload.items()}


def save_calorie_map(mapping: dict[str, int], data_dir: str | Path | None = None) -> Path:
    path = calorie_map_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {normalize_label(key): int(value) for key, value in mapping.items()}

    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(sorted(normalized.items())), handle, indent=2)

    return path


def save_class_names(class_names: list[str], destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for class_name in class_names:
            handle.write(f"{normalize_label(class_name)}\n")
    return path


def load_class_names(source: str | Path) -> list[str]:
    path = Path(source)
    with path.open("r", encoding="utf-8") as handle:
        return [normalize_label(line) for line in handle.read().splitlines() if line.strip()]
