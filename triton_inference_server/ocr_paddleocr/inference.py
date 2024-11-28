import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *


def query_triton_server(image_paths: list[str], url: str = "localhost:8000"):
    """
    Query Triton server with a list of image paths
    """
    # Create client
    client = httpclient.InferenceServerClient(url=url)

    # Prepare input data as a single batch
    input_data = np.array(
        [path.encode("utf-8") for path in image_paths], dtype=np.object_
    )

    # Create input tensor
    inputs = [
        httpclient.InferInput(
            "input_texts",
            [len(image_paths)],
            datatype="BYTES",
        )
    ]
    inputs[0].set_data_from_numpy(input_data)

    # Create output tensor
    outputs = [httpclient.InferRequestedOutput("output_texts", binary_data=False)]

    # Perform inference
    response = client.infer(model_name="ppocrv4", inputs=inputs, outputs=outputs)

    output_data = response.as_numpy("output_texts")

    import json

    results = [json.loads(detection) for detection in output_data]

    return results


# Example usage
if __name__ == "__main__":
    images = [
        "https://raw.githubusercontent.com/dnth/cv-docker-images/refs/heads/main/ocr/test_images/test_image_3.jpg",
        "https://raw.githubusercontent.com/dnth/cv-docker-images/refs/heads/main/ocr/test_images/test_image_1.jpg",
    ]

    try:
        results = query_triton_server(images)
        print(results)

        for img_path, detections in zip(images, results):
            print(f"\nResults for {img_path}:")
            for detection in detections:
                print(f"Text: {detection['text']}")
                print(f"Confidence: {detection['confidence']}")
                print(f"BBox: {detection['bbox']}")
                print("---")

    except Exception as e:
        print(f"Error: {e}")
