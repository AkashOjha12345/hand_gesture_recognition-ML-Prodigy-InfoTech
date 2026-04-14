import argparse

from src.calorie_lookup import build_calorie_map
from src.predict import main as predict_main
from src.preprocess import (
    class_names_from_layout,
    dataset_layout,
    default_data_dir,
    ensure_project_structure,
    save_calorie_map,
)
from src.train_model import main as train_main
from src.webcam import main as webcam_main


def init_calorie_map(data_dir: str) -> None:
    layout = dataset_layout(data_dir)
    class_names = class_names_from_layout(layout)
    destination = save_calorie_map(build_calorie_map(class_names), data_dir)
    print(f"Calorie map created at: {destination}")


def main() -> None:
    ensure_project_structure()

    parser = argparse.ArgumentParser(
        description="Food recognition and calorie estimation toolkit."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the food classifier.")
    train_parser.add_argument("--data-dir", default=str(default_data_dir()))
    train_parser.add_argument("--output-dir")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--seed", type=int, default=42)

    predict_parser = subparsers.add_parser("predict", help="Predict the food item for an image.")
    predict_parser.add_argument("image")
    predict_parser.add_argument("--models-dir")
    predict_parser.add_argument("--data-dir", default=str(default_data_dir()))
    predict_parser.add_argument("--top-k", type=int, default=3)

    webcam_parser = subparsers.add_parser("webcam", help="Run live webcam inference.")
    webcam_parser.add_argument("--camera-index", type=int, default=0)
    webcam_parser.add_argument("--models-dir")
    webcam_parser.add_argument("--data-dir", default=str(default_data_dir()))

    init_parser = subparsers.add_parser(
        "init-calories",
        help="Create a calorie map template from the discovered dataset classes.",
    )
    init_parser.add_argument("--data-dir", default=str(default_data_dir()))

    args = parser.parse_args()

    if args.command == "init-calories":
        init_calorie_map(args.data_dir)
        return

    if args.command == "train":
        import sys

        sys.argv = [
            "train",
            "--data-dir",
            args.data_dir,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
        ]
        if args.output_dir:
            sys.argv.extend(["--output-dir", args.output_dir])
        train_main()
        return

    if args.command == "predict":
        import sys

        sys.argv = [
            "predict",
            args.image,
            "--data-dir",
            args.data_dir,
            "--top-k",
            str(args.top_k),
        ]
        if args.models_dir:
            sys.argv.extend(["--models-dir", args.models_dir])
        predict_main()
        return

    if args.command == "webcam":
        import sys

        sys.argv = [
            "webcam",
            "--camera-index",
            str(args.camera_index),
            "--data-dir",
            args.data_dir,
        ]
        if args.models_dir:
            sys.argv.extend(["--models-dir", args.models_dir])
        webcam_main()


if __name__ == "__main__":
    main()
