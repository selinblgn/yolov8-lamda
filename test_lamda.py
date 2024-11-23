import requests
import base64

# Local test image path
IMG_PATH = '25756.jpg'
API_URL = 'https://yasnwgtl13.execute-api.eu-central-1.amazonaws.com/predict'

# Encode the image to base64
with open(IMG_PATH, 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode('ascii')

# Send the request to Lambda
payload = {
    "image": img_b64,
    "conf_thres": 0.3  # Optional: Confidence threshold
}

response = requests.post(API_URL, json=payload)

# Extract and print detections
detections = response.json().get('detections', [])
print(detections)
