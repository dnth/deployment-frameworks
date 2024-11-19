# Model Deployment Frameworks
This repo provides a simple example of how to deploy a model using different frameworks.

## Triton Inference Server
Export the model into the `model_repository` directory.

```bash
cd triton_inference_server/image_classification_resnet50
```

```bash
python export_tv_model.py
```


Create a config.pbtxt file in the `model_repository/resnet50/` directory with the following content:

```pbtxt
name: "resnet50"
backend: "pytorch"
max_batch_size: 8
input [
  {
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]
output [
  {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]
instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

Run the Triton Inference Server with the ResNet50 model.

```bash
cd triton_inference_server/image_classification_resnet50
```

```bash
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:23.01-py3 tritonserver --model-repository=/models
```

Query the model.

```bash
cd triton_inference_server/image_classification_resnet50
```

```bash
python query.py
```

## Ray Serve
Launch the Ray application.

```bash
cd ray_serve
```

```bash
python deploy.py
```

Query the model.

```bash
cd ray_serve
```

```bash
python query.py
```

## Ray + Triton

