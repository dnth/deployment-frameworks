# Local Run
Run PaddleOCR on GPU locally using Triton Inference Server.

## Build Docker Image

```bash
cd triton_inference_server/ocr_paddleocr
```

```bash
docker build -t triton-paddleocr . 
```

## Run Docker Container

```bash
docker run --gpus all --rm \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    -v ${PWD}/model_repository:/models \
    triton-paddleocr \
    tritonserver --model-repository=/models
```

## Query Triton Server

```bash
python inference.py
```