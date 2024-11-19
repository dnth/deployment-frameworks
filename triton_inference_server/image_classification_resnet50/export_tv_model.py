import os

import torch
import torchvision.models as models

# Load a pretrained ResNet50 model
model = models.resnet50(pretrained=True)
model.eval()

# Create an example input
example_input = torch.randn(1, 3, 224, 224)

# Create directory if it doesn't exist
os.makedirs("model_repository/resnet50/1", exist_ok=True)

# Export the model to TorchScript
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model_repository/resnet50/1/model.pt")
