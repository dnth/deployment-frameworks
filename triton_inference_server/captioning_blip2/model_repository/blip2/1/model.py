import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor


class TritonPythonModel:
    def initialize(self, args):
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b"
        )
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device, dtype=torch.bfloat16)

    def execute(self, requests):
        responses = []
        for request in requests:
            # Get the input tensor
            input_tensor = request.inputs[0].as_numpy()

            # The input tensor already includes batch dimension, so we take the first item
            # since we're processing one request at a time
            image_data = input_tensor[0]  # Remove batch dimension for PIL

            # Convert to PIL Image
            pil_image = Image.fromarray(image_data.astype("uint8"))

            # Process image
            inputs = self.processor(images=pil_image, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(self.device)

            # Generate caption
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
                generated_text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

            # Create output tensor
            output_tensor = pb_utils.Tensor(
                "output", np.array(generated_text, dtype=np.object_)
            )

            # Create and append InferenceResponse
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )
            responses.append(inference_response)

        return responses
