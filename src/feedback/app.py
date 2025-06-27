import os
from typing import TypedDict

from flask import Flask, g
from flask_restx import Api, Resource, reqparse
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://mongo:27017")
DATABASE_NAME = os.environ.get("DATABASE_NAME", "feedback_db")


def get_db():
    if "db" not in g:
        g.mongoclient = MongoClient(f"{MONGODB_URI}/{DATABASE_NAME}")

        try:
            g.mongoclient.admin.command("ping")
            print("Successfully connected to MongoDB!")
        except ConnectionFailure:
            print("MongoDB not available!!!")

        g.db = g.mongoclient[DATABASE_NAME]

    return g.db


app = Flask(__name__)
api = Api(app)


@app.teardown_appcontext
def close_db_connection(exc):
    mongoclient = g.pop("mongoclient", None)
    if mongoclient is not None:
        mongoclient.close()


class Detection(TypedDict):
    label: str
    boxes: list[float]


def detections(value: list[Detection]):
    """Validate and parse a list of detection objects from the request body.

    This function ensures that each detection object in the list contains
    the required 'label' and 'boxes' keys before returning the validated list.

    Args:
        value (list[Detection]): A list of detection objects to validate.
            Each detection must contain:
            - 'label': A string identifying the detected object (e.g., 'cat', 'dog')
            - 'boxes': A list of 4 float values representing the bounding box coordinates
              in the format [x1, y1, x2, y2]

    Returns:
        list[Detection]: The validated list of detection objects

    Raises:
        ValueError: If any detection is missing the required 'label' or 'boxes' field
    """
    for v in value:
        if "label" not in v:
            raise ValueError("label is required")
        if "boxes" not in v:
            raise ValueError("boxes is required")
    return value


feedback_parser = reqparse.RequestParser()
feedback_parser.add_argument(
    "image_id", type=str, required=True, help="The id used to reference the image that needs corrections"
)
feedback_parser.add_argument(
    "detections",
    required=True,
    type=detections,
    location="json",
    help="A list of detections that need to be corrected",
)


@api.route("/feedback")
class Feedback(Resource):
    def post(self):
        args = feedback_parser.parse_args()

        collection = get_db().feedback
        result = collection.insert_one(args)

        return {"message": "Feedback received successfully", "_id": str(result.inserted_id)}, 201


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
