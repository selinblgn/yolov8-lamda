import boto3
import numpy as np
from PIL import Image
import onnxruntime as ort
import io
import os
import json
import logging
import base64

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_image(image_data, input_shape):
    """
    Preprocess the image to match the ONNX model input.
    Converts the image to RGB, resizes it, normalizes pixel values,
    and reshapes it to the model's input dimensions.
    """
    try:
        logger.info("Starting image preprocessing.")
        image = Image.open(image_data).convert("RGB")
        image = image.resize((input_shape[2], input_shape[3]))
        image_array = np.asarray(image).astype('float32') / 255.0  # Normalize to [0, 1]
        image_array = np.transpose(image_array, [2, 0, 1])  # Change to channels-first format
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        logger.info("Image preprocessing completed.")
        return image_array
    except Exception as e:
        logger.error(f"Error during image preprocessing: {e}")
        raise

def load_model(onnx_model_path):
    """
    Loads the ONNX model from the specified path.
    """
    try:
        logger.info(f"Loading ONNX model from path: {onnx_model_path}")
        session = ort.InferenceSession(onnx_model_path)
        logger.info("ONNX model loaded successfully.")
        return session
    except Exception as e:
        logger.error(f"Error loading ONNX model: {e}")
        raise

def process_predictions(predictions, conf_threshold=0.3):
    """
    Process the raw predictions from the ONNX model and return the formatted output.
    Filters predictions by the confidence threshold.
    """
    try:
        logger.info("Processing predictions.")
        boxes, scores, class_ids = predictions
        detections = []

        # Iterate over each prediction and format it
        for i in range(len(scores)):
            if scores[i] >= conf_threshold:
                detection = {
                    "bbox": boxes[i].tolist(),  # Convert box to list
                    "score": scores[i],        # Confidence score
                    "class_id": int(class_ids[i])  # Convert class_id to int
                }
                detections.append(detection)
        
        logger.info("Prediction processing completed.")
        return detections
    except Exception as e:
        logger.error(f"Error processing predictions: {e}")
        raise

def run_inference(session, input_data):
    """
    Runs inference on the input data using the ONNX model.
    """
    try:
        logger.info("Running inference.")
        input_name = session.get_inputs()[0].name
        predictions = session.run(None, {input_name: input_data})
        logger.info("Inference completed.")
        
        # Process predictions into the desired format
        detections = process_predictions(predictions)
        return detections
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise

def handler(event, context):
    """
    AWS Lambda handler function to process a base64-encoded image,
    run inference using an ONNX model, and return predictions.
    """
    try:
        logger.info(f"Received event: {json.dumps(event)}")

        # Parse the request body
        body = json.loads(event['body'])
        img_b64 = body['image']
        conf_thres = body.get('conf_thres', 0.3)
        
        # Decode the base64-encoded image
        image_data = io.BytesIO(base64.b64decode(img_b64))

        # Load the ONNX model
        onnx_model_path = os.environ.get('ONNX_MODEL_PATH', '/opt/model/best.onnx')
        session = load_model(onnx_model_path)

        # Adjust input shape for preprocessing
        input_shape = session.get_inputs()[0].shape
        input_shape = [int(dim) if isinstance(dim, (int, float)) else 1 for dim in input_shape]

        # Preprocess the image
        input_data = preprocess_image(image_data, input_shape)

        # Ensure input data is the correct type
        input_data = input_data.astype(np.float32)

        # Perform inference and get formatted predictions
        detections = run_inference(session, input_data)

        # Return predictions as JSON
        return {
            "statusCode": 200,
            "body": json.dumps({
                "detections": detections
            })
        }

    except Exception as e:
        logger.error(f"Handler encountered an error: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
