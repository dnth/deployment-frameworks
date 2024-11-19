import numpy as np
import torchvision.transforms as transforms
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.utils import triton_to_np_dtype


# Preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch.numpy()


# Set up the client
client = httpclient.InferenceServerClient(url="localhost:8000")

# Prepare the input data
input_data = preprocess_image(
    "/home/dnth/Desktop/xinfer/assets/demo/0a6ee446579d2885.jpg"
)

# Set up the input
inputs = [httpclient.InferInput("input__0", input_data.shape, datatype="FP32")]
inputs[0].set_data_from_numpy(input_data, binary_data=True)

# Set up the output
outputs = [httpclient.InferRequestedOutput("output__0", binary_data=True)]

# Send inference request
results = client.infer(model_name="resnet50", inputs=inputs, outputs=outputs)

# Get and process the results
output = results.as_numpy("output__0")
predicted_class = np.argmax(output)
print(f"Predicted class: {predicted_class}")
