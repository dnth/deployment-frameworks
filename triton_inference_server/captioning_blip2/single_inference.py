import numpy as np
import tritonclient.http as httpclient
from PIL import Image

client = httpclient.InferenceServerClient(url="localhost:8000")

# Load image and convert to RGB to ensure 3 channels
image = Image.open(
    "/home/dnth/Desktop/xinfer/assets/demo/0a6ee446579d2885.jpg"
).convert("RGB")
input_data = np.array(image).astype(np.uint8)  # Changed to uint8

# Add batch dimension to input_data
input_data = np.expand_dims(input_data, axis=0)  # Shape becomes (1, height, width, 3)
print(input_data.shape)

inputs = [httpclient.InferInput("input", input_data.shape, "UINT8")]  # Changed to UINT8
inputs[0].set_data_from_numpy(input_data)

outputs = [httpclient.InferRequestedOutput("output", binary_data=False)]

response = client.infer("blip2", inputs, outputs=outputs)
output = response.as_numpy("output")
print("Generated caption:", output[0])
