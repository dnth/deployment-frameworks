#!/bin/bash

requirements=(
    torch
    torchvision
    transformers
)

cat <<EOF | docker build --network host --tag my-tritonserver -
FROM nvcr.io/nvidia/tritonserver:24.09-py3
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir ${requirements[@]}
EOF