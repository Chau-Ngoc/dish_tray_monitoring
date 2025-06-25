import argparse
import os
import sys
from pathlib import Path

import cv2 as cv
import supervision as sv

START_X, START_Y = 950, 70
ROI_WIDTH, ROI_HEIGHT = 580, 180


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run YOLO inference on images")

    # Model arguments
    parser.add_argument("--model", "-m", type=str, required=True, help="Path to YOLO model file (.pt, .onnx, .engine)")
    parser.add_argument(
        "--persist",
        action="store_true",
        help="This argument tells the tracker that the current image or frame is the next in a sequence and to expect tracks from the previous image in the current image",
    )
    parser.add_argument("--tracker", type=str, default="botsort.yaml", help="The tracking algorithm to use")

    # Input arguments
    parser.add_argument("--source", "-s", type=str, required=True, help="The camera footage to monitor")

    # Output arguments
    parser.add_argument("--save-txt", action="store_true", help="Save results as txt files")
    parser.add_argument("--save-conf", action="store_true", help="Save confidence scores in txt files")
    parser.add_argument("--save-crop", action="store_true", help="Save cropped detection images")
    parser.add_argument("--nosave", action="store_true", help="Do not save images/videos")

    # Detection parameters
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IoU threshold for NMS (default: 0.45)")
    parser.add_argument("--max-det", type=int, default=1000, help="Maximum detections per image (default: 1000)")
    parser.add_argument("--classes", nargs="+", type=int, help="Filter by class: --classes 0 2 3")

    # Image processing
    parser.add_argument("--imgsz", "--img-size", type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument("--device", type=str, default="", help="Device to run on: cpu, 0, 1, etc. (default: auto)")

    # Display options
    parser.add_argument("--view-img", action="store_true", help="Display results in window")
    parser.add_argument("--hide-labels", action="store_true", help="Hide labels in output")
    parser.add_argument("--hide-conf", action="store_true", help="Hide confidence scores in output")
    parser.add_argument("--line-thickness", type=int, default=3, help="Bounding box thickness (default: 3)")

    # Miscellaneous
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--project", type=str, help="Project directory")
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--exist-ok", action="store_true", help="Existing project/name ok, do not increment")

    return parser.parse_args()


def load_model(model_path):
    """Load YOLO model."""
    try:
        from ultralytics import YOLO

        model = YOLO(model_path)
        return model
    except ImportError:
        print("Error: ultralytics package not found. Install with: pip install ultralytics")
        return None


def resize_roi_to_orig_size(roi, fx, fy):
    """Resize region-of-interest area to its original size"""
    target_width = int(640 / fx)
    target_height = int(640 / fy)
    return cv.resize(roi, (target_width, target_height), None, cv.INTER_LINEAR)


def run_monitor(model, source, args):
    """Run inference using Ultralytics YOLO."""
    cap = cv.VideoCapture(source)

    fourcc = cv.VideoWriter_fourcc(*"XVID")
    outfile = os.environ.get("OUTPUT_DIR", "runs") + "/output.mp4"
    ow, oh = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    wrt = cv.VideoWriter(outfile, fourcc, fps, (ow, oh))

    annotator = sv.BoxAnnotator()

    fy = 640 / ROI_HEIGHT
    fx = 640 / ROI_WIDTH

    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                roi = frame.copy()[START_Y : START_Y + ROI_HEIGHT, START_X : START_X + ROI_WIDTH]
                roi = cv.resize(roi, None, None, fx, fy, cv.INTER_LINEAR)

                results = model.track(
                    source=roi,
                    persist=args.persist,
                    tracker=args.tracker,
                    conf=args.conf_thres,
                    iou=args.iou_thres,
                    max_det=args.max_det,
                    classes=args.classes,
                    imgsz=args.imgsz,
                    device=args.device,
                    save=not args.nosave,
                    save_txt=args.save_txt,
                    save_conf=args.save_conf,
                    save_crop=args.save_crop,
                    show=args.view_img,
                    project=args.project,
                    name=args.name,
                    exist_ok=args.exist_ok,
                    line_width=args.line_thickness,
                    show_labels=not args.hide_labels,
                    show_conf=not args.hide_conf,
                    verbose=args.verbose,
                )[0]

                detections = sv.Detections.from_ultralytics(results)
                annotated_roi = annotator.annotate(roi, detections)
                annotated_roi = resize_roi_to_orig_size(annotated_roi, fx, fy)
                cv.rectangle(annotated_roi, (0, 0), (ROI_WIDTH, ROI_HEIGHT), (0, 0, 255), 2)
                frame[START_Y : START_Y + ROI_HEIGHT, START_X : START_X + ROI_WIDTH] = annotated_roi
                wrt.write(frame)
    except KeyboardInterrupt:
        print("Process interrupted by user.")
    finally:
        cap.release()
        wrt.release()


def validate_inputs(args):
    """Validate input arguments."""
    # Check if source exists (unless it's a camera index)
    if not args.source.isdigit():
        if not os.path.exists(args.source):
            print(f"Error: Source '{args.source}' not found.")
            return False

    return True


def main():
    """Main function to run YOLO inference."""
    args = parse_arguments()

    if not validate_inputs(args):
        sys.exit(1)

    if args.verbose:
        print(f"Arguments: {vars(args)}")

    # Detect YOLO type

    # Load model
    model = load_model(args.model)

    if model is None:
        print("Failed to load model. Please check your model file and dependencies.")
        sys.exit(1)

    print(f"Model loaded successfully: {args.model}")

    try:
        # Run inference
        run_monitor(model, args.source, args)

        print("Inference completed successfully!")

        if args.verbose:
            print(f"Results saved to: {Path(args.project) / args.name}")

    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
