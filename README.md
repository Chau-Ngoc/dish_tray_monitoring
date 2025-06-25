# Object Detection and Tracking with YOLO

This project implements object detection and tracking using the YOLO (You Only Look Once) model.
It allows you to train a custom YOLO model, track objects in videos, and collect user feedback.

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Set up Docker:
   ```
   docker compose build
   ```

This will build the Docker images and install all required dependencies.
It also creates three bind mounts:
- `./data:/app/data`: Directory for storing training data.
- `./runs:/app/runs`: Directory for storing model checkpoints and training logs.
  Additionally, this directory is used for storing the output of the tracking process.
- `./models:/app/models`: Directory for storing trained models.
That's it. You're ready to go!

## Training the YOLO Model

To train a custom YOLO model on your dataset:

1. Put your data in the `data/` directory mentioned above. Prepare your dataset in the following structure:
   ```
   data/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── val/
   │   ├── images/
   │   └── labels/
   └── data.yaml
   ```
   The `data.yaml` file should follow [Ultralytics YOLO format](https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format)

   I have included a sample dataset in the `data/` directory. You can use it or create your own.

2. Start the training process:
   ```
   docker compose run --rm monitor-service train.py --model <model_name> --data <path_to_data.yaml> --epochs 10 --batch-size 16
   ```

   **Note:** The `path_to_data.yaml` argument should point to the directory inside the docker container, not to the host machine. In
   my setup, it's `/app/data/`.

   To view all available training options, run:
   ```
   docker compose run --rm monitor-service train.py --help
   ```

3. The trained weights will be saved in the `runs/train/weights/` directory.

## Track the dishes and trays in videos

I have provided a sample video for you to play with in the `data/` directory. To track objects in an input video:

1. Ensure you have a trained model inside the `model/` directory or use a pre-trained YOLO model.

2. Run the tracking command:
   ```
   docker compose run --rm monitor-service monitor.py --model <path_to_best_model> --source <path_to_video.mp4>
   ```

   **Note:** The `path_to_best_model` and `path_to_video.mp4` arguments should point to the directory inside the docker container, not to the host machine. In
   my setup, it's `/app/models/` and `/app/data/example_video_1.mp4` respectively..

   To view all available options, run:
    ```
   docker compose run --rm monitor-service monitor.py --help
   ```

3. The processed video will be saved in the `runs/` directory.

## Feedback API

This project includes a feedback API to collect user feedback on detection and tracking results.
The feedback is then stored in MongoDB for further analysis. This will allow you to retrain the model on the feedback data to
improve the model's performance.

### Starting the Feedback Server

Run

```shell
docker compose up feedback-service
```

The feedback server will be available at `http://0.0.0.0:5000/feedback`.

### Submitting Feedback

Simply send a POST request to the endpoint `http://0.0.0.0:5000/feedback` with `Content-Type: application/json`.
The request body should be a JSON object with the following fields:

```json
{
    "image_id": "001.jpg",
    "detections": [
        {"label": "cat", "boxes": [120, 123, 5, 7]},
        {"label": "cat", "boxes": [120, 123, 7, 7]}
    ]
}
```
* `image_id` is a string and it refers to the filename of the image in the dataset.
* `detections` is a list of detections in the image. Each detection is a dictionary with the following fields:
    * `label`: The label of the detection. e.g. 'cat', 'dog', 'person'
    * `boxes`: A list of bounding boxes for the detection. Each box is a list of four numbers: `[x1, y1, x2, y2]`.`
