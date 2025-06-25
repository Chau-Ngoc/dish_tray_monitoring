import argparse
import os
import sys
from pathlib import Path


def parse_arguments():
    """Parse command line arguments for YOLO training."""
    parser = argparse.ArgumentParser(
        description="Train YOLO model with custom parameters", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument("--data", type=str, required=True, help="Path to dataset YAML file (e.g., data.yaml)")

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Model to train (For example: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)",
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")

    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")

    parser.add_argument("--img-size", "--imgsz", type=int, default=640, help="Image size for training (pixels)")

    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")

    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay for optimizer")

    # Output and project settings
    parser.add_argument("--project", type=str, default="runs", help="Project directory to save results")

    parser.add_argument("--name", type=str, help="Experiment name")

    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every n epochs (-1 to disable)")

    # Device and performance
    parser.add_argument("--device", type=str, default="", help="Device to run on (cpu, 0, 1, 2, 3, etc.)")

    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads for data loading")

    # Training options
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")

    parser.add_argument(
        "--cache", choices=["ram", "disk", None], default=None, help="Cache images for faster training (ram/disk)"
    )

    parser.add_argument("--single-cls", action="store_true", help="Train as single-class dataset")

    parser.add_argument("--optimizer", choices=["SGD", "Adam", "AdamW"], default="SGD", help="Optimizer to use")

    parser.add_argument(
        "--patience", type=int, default=50, help="Epochs to wait for no improvement before early stopping"
    )

    # Validation and testing
    parser.add_argument("--val", action="store_true", default=True, help="Validate during training")

    parser.add_argument("--save-json", action="store_true", help="Save results to JSON file")

    # Advanced options
    parser.add_argument("--cfg", type=str, default="", help="Model configuration file")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")

    return parser.parse_args()


def validate_args(args):
    """Validate command line arguments."""
    # Check if data file exists
    if not Path(args.data).exists():
        print(f"Error: Dataset file '{args.data}' not found!")
        sys.exit(1)

    # Validate epochs
    if args.epochs <= 0:
        print("Error: Epochs must be greater than 0!")
        sys.exit(1)

    # Validate batch size
    if args.batch_size <= 0:
        print("Error: Batch size must be greater than 0!")
        sys.exit(1)

    # Validate image size
    if args.img_size <= 0:
        print("Error: Image size must be greater than 0!")
        sys.exit(1)

    print("✓ Arguments validated successfully!")


def train_yolo(args):
    """Train YOLO model with given arguments."""
    try:
        # Import YOLO (assuming ultralytics)
        from ultralytics import YOLO

        # Load model
        print(f"Loading model: {args.model}")
        model = YOLO(args.model)

        # Prepare training arguments
        train_args = {
            "data": args.data,
            "epochs": args.epochs,
            "batch": args.batch_size,
            "imgsz": args.img_size,
            "lr0": args.lr0,
            "weight_decay": args.weight_decay,
            "project": args.project,
            "name": args.name,
            "workers": args.workers,
            "optimizer": args.optimizer,
            "patience": args.patience,
            "val": args.val,
            "save_json": args.save_json,
            "verbose": args.verbose,
            "seed": args.seed,
        }

        # Add optional arguments
        if args.device:
            train_args["device"] = args.device
        if args.cache:
            train_args["cache"] = args.cache
        if args.single_cls:
            train_args["single_cls"] = args.single_cls
        if args.resume:
            train_args["resume"] = args.resume
        if args.save_period > 0:
            train_args["save_period"] = args.save_period
        if args.cfg:
            train_args["cfg"] = args.cfg

        print("Starting YOLO training with the following parameters:")
        for key, value in train_args.items():
            print(f"  {key}: {value}")

        # Start training
        model.train(**train_args)

        print("Training completed successfully!")

    except ImportError:
        print("Error: ultralytics package not found!")
        print("Install it with: pip install ultralytics")
        sys.exit(1)
    except Exception as e:
        print(f"Error during training: {str(e)}")
        sys.exit(1)


def main():
    """Main function to run YOLO training."""
    print("YOLO Training Script")
    print("=" * 50)

    # Parse arguments
    args = parse_arguments()

    # Display parsed arguments
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    # Validate arguments
    validate_args(args)

    # Create output directory
    os.makedirs(args.project, exist_ok=True)

    # Start training
    train_yolo(args)


if __name__ == "__main__":
    main()
