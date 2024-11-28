import json
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
import triton_python_backend_utils as pb_utils
from paddleocr import PaddleOCR


class TritonPythonModel:
    def initialize(self, args):
        """Initialize the model"""
        self.model = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            use_gpu=True,
            ocr_version="PP-OCRv4",
            show_log=True,
        )

    def infer(self, image_path: str) -> list[dict]:
        """Perform OCR on an image and return detected text with positions"""
        # Check if the path is a URL
        if urlparse(image_path).scheme in ("http", "https"):
            response = requests.get(image_path)
            response.raise_for_status()
            image_array = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Failed to load image from path: {image_path}")

        ocr_results = self.model.ocr(image)
        detected_regions = []

        if ocr_results == [None]:
            return detected_regions

        for text_regions in ocr_results:
            for region in text_regions:
                bbox_points, (text, confidence) = region
                points_array = np.array(bbox_points, dtype=np.float32)
                x, y, w, h = map(int, cv2.boundingRect(points_array))

                detected_regions.append(
                    {
                        "text": text,
                        "confidence": confidence,
                        "bbox": {"x": x, "y": y, "width": w, "height": h},
                    }
                )

        return detected_regions

    def execute(self, requests):
        """Handle inference requests"""
        responses = []

        for request in requests:
            # Get the input tensor
            input_tensor = pb_utils.get_input_tensor_by_name(request, "input_texts")
            image_paths = [path.decode("utf-8") for path in input_tensor.as_numpy()]

            # Process each image
            batch_results = [self.infer(image_path) for image_path in image_paths]

            # Convert results to JSON strings
            output_data = np.array(
                [json.dumps(result).encode("utf-8") for result in batch_results],
                dtype=np.object_,
            )

            # Create output tensor
            output_tensor = pb_utils.Tensor("output_texts", output_data)

            # Create and append the inference response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        """Clean up when the model is unloaded"""
        pass
