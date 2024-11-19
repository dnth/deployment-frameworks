from pathlib import Path

import requests

# Replace with your image file path
image_path = Path("/home/dnth/Desktop/xinfer/assets/demo/0a6ee446579d2885.jpg")

# Read the image file as binary data
image_bytes = image_path.read_bytes()

# Make POST request to the service
# By default, Ray Serve runs on port 8000
response = requests.post("http://localhost:8000/", data=image_bytes)

# Print the predictions
print(response.json())
