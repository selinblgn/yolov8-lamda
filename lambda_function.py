import boto3
import numpy as np
from PIL import Image
import onnxruntime as ort
import io
import os
import json
import logging
import base64
import cv2
   
# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the S3 client
s3 = boto3.client('s3')

def preprocess_image(image_data, input_shape):
    """
    Preprocess the image to match the ONNX model input.
    """
    try:
        logger.info("Starting image preprocessing.")
        image = Image.open(image_data).convert("RGB")
        image = image.resize((input_shape[2], input_shape[3]))
        image_array = np.asarray(image).astype('float32') / 255.0  # Normalize
        image_array = np.transpose(image_array, [2, 0, 1])  # Channels-first
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        logger.info("Image preprocessing completed.")
        return image_array
    except Exception as e:
        logger.error(f"Error during image preprocessing: {e}")
        raise

def download_file_from_s3(bucket_name, key):
    """
    Downloads a file from an S3 bucket.
    """
    try:
        logger.info(f"Downloading file from S3: bucket={bucket_name}, key={key}")
        file_data = io.BytesIO()
        s3.download_fileobj(bucket_name, key, file_data)
        file_data.seek(0)
        logger.info("File downloaded successfully.")
        return file_data
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '403':
            logger.error("Access denied to the S3 object.")
            raise PermissionError("Access denied to the S3 object.")
        else:
            logger.error(f"Error downloading file from S3: {e}")
            raise
    except Exception as e:
        logger.error(f"Error downloading file from S3: {e}")
        raise

def load_model(onnx_model_data):
    """
    Loads the ONNX model from a byte stream.
    """
    try:
        logger.info("Loading ONNX model.")
        session = ort.InferenceSession(onnx_model_data.read())
        logger.info("ONNX model loaded successfully.")
        return session
    except Exception as e:
        logger.error(f"Error loading ONNX model: {e}")
        raise

def process_predictions(predictions, conf_threshold=0.3):
    """
    Process the raw predictions from the ONNX model.
    """
    try:
        logger.info("Processing predictions.")
        if len(predictions) == 1:
            predictions = predictions[0]
        boxes, scores, class_ids = predictions[0], predictions[1], predictions[2]
        detections = []

        for i in range(len(scores)):
            if scores[i] >= conf_threshold:
                detection = {
                    "bbox": [int(coord) for coord in boxes[i]],
                    "score": round(float(scores[i]), 3),
                    "class_id": int(class_ids[i])
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
        return process_predictions(predictions)
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise

def handler(event, context):
    """
    AWS Lambda handler function.
    """
    try:
        logger.info(f"Received event: {json.dumps(event)}")

        bucket_name = os.environ.get('S3_BUCKET_NAME')
        image_key = event.get('queryStringParameters', {}).get('image_name')
        model_key = os.environ.get('S3_MODEL_KEY', 'best.onnx')

        if not bucket_name or not image_key:
            logger.error("Missing bucket name or image key.")
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing bucket name or image key"})
            }

        # Download model and image from S3
        model_data = download_file_from_s3(bucket_name, model_key)
        image_data = download_file_from_s3(bucket_name, image_key)

        # Load model
        session = load_model(model_data)
        input_shape = session.get_inputs()[0].shape
        input_shape = [int(dim) if isinstance(dim, (int, float)) else 1 for dim in input_shape]

        # Preprocess image and run inference
        input_data = preprocess_image(image_data, input_shape)
        detections = run_inference(session, input_data)

        return {
            "statusCode": 200,
            "body": json.dumps({"detections": detections})
        }

    except Exception as e:
        logger.error(f"Handler encountered an error: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
              