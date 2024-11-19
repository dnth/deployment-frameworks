import io
from typing import Dict

import requests
import torch
from PIL import Image
from ray import serve
from starlette.requests import Request
from torchvision import models, transforms


@serve.deployment
class ResNetDeployment:
    def __init__(self):
        # Initialize ResNet50 model
        self.model = models.resnet50(pretrained=True)
        self.model.eval()

        # Standard ImageNet transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Load ImageNet class labels
        with open("imagenet_classes.txt") as f:
            self.categories = [line.strip() for line in f.readlines()]

    async def __call__(self, request: Request) -> Dict:
        # Get image data from request
        image_payload = await request.body()
        image = Image.open(io.BytesIO(image_payload))

        # Preprocess image
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        # Get prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get top 5 predictions
        top5_prob, top5_catid = torch.topk(probabilities, 5)

        return {
            "predictions": [
                {"category": self.categories[catid.item()], "probability": prob.item()}
                for prob, catid in zip(top5_prob, top5_catid)
            ]
        }


app = ResNetDeployment.bind()

# Deploy the application
serve.run(app, route_prefix="/", blocking=True)
