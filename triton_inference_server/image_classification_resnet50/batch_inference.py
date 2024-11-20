import time

import numpy as np
import torchvision.transforms as transforms
import tritonclient.http as httpclient
from PIL import Image


def load_imagenet_classes(path="imagenet_classes.txt"):
    with open(path) as f:
        categories = [s.strip() for s in f.readlines()]
    return categories


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


# New function to handle multiple images
def preprocess_images(image_paths):
    batch = []
    for image_path in image_paths:
        processed = preprocess_image(image_path)
        batch.append(processed[0])  # Remove the batch dimension since we'll stack them
    return np.stack(batch)


# Load ImageNet classes and get the class name
classes = load_imagenet_classes()

# Set up the client
client = httpclient.InferenceServerClient(url="localhost:8000")

# Example list of image paths
image_paths = [
    "/home/dnth/Desktop/x.retrieval/nbs/data/coco/val2017/000000000139.jpg",
    "/home/dnth/Desktop/x.retrieval/nbs/data/coco/val2017/000000000285.jpg",
    "/home/dnth/Desktop/x.retrieval/nbs/data/coco/val2017/000000000632.jpg",
    "/home/dnth/Desktop/x.retrieval/nbs/data/coco/val2017/000000000724.jpg",
    "/home/dnth/Desktop/x.retrieval/nbs/data/coco/val2017/000000000776.jpg",
    "/home/dnth/Desktop/x.retrieval/nbs/data/coco/val2017/000000000785.jpg",
    "/home/dnth/Desktop/x.retrieval/nbs/data/coco/val2017/000000000802.jpg",
    "/home/dnth/Desktop/x.retrieval/nbs/data/coco/val2017/000000000872.jpg",
    "/home/dnth/Desktop/x.retrieval/nbs/data/coco/val2017/000000000885.jpg",
    "/home/dnth/Desktop/x.retrieval/nbs/data/coco/val2017/000000001000.jpg",
    # Add more paths as needed
]

# Prepare batch input
start_time = time.time()

input_data = preprocess_images(image_paths)

# Set up the input
inputs = [httpclient.InferInput("input__0", input_data.shape, datatype="FP32")]
inputs[0].set_data_from_numpy(input_data, binary_data=True)

# Set up the output
outputs = [httpclient.InferRequestedOutput("output__0", binary_data=True)]

# Send inference request
results = client.infer(model_name="resnet50", inputs=inputs, outputs=outputs)

end_time = time.time()
inference_time = end_time - start_time

# Process batch results
output = results.as_numpy("output__0")
predicted_idx = np.argmax(output, axis=1)  # Note the axis=1 for batch processing

# Print results for each image
for i, idx in enumerate(predicted_idx):
    predicted_class = classes[idx]
    print(f"Image {i+1} - Predicted class: {predicted_class} (index: {idx})")

# Print timing information
print(f"\nTotal time taken: {inference_time*1000:.3f} ms")
print(f"Average time per image: {inference_time/len(image_paths)*1000:.3f} ms")
