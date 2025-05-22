# Dockerfile for building the OCR image with a pre-downloaded fine-tuned recognition model

# Base image: NVIDIA CUDA image
# For paddlepaddle-gpu==3.0.0 (which uses CUDA 12.x libraries) and H100 (Compute Capability 9.0)
# CUDA 12.2.2 and cuDNN 8 provides a compatible environment.
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

LABEL maintainer="your_name_or_email"
LABEL description="OCR Engine with PaddleOCR, FastAPI, GPU, pre-downloaded base models, and a custom fine-tuned recognition model."

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV DEBIAN_FRONTEND=noninteractive

# --- Environment Variables for Model Paths ---
# Base directory for all models within the image
ENV MODELS_BASE_DIR=/opt/paddleocr_models

# Specific model directory names (these are the names of the folders AFTER extraction/copying)
ENV DET_MODEL_SUBDIR=det/en/en_PP-OCRv3_det_infer
# Use a descriptive name for your fine-tuned model's directory within the image
ENV FINETUNED_REC_MODEL_DIR_NAME=my_custom_finetuned_ocrv4_rec
ENV REC_MODEL_SUBDIR=rec/en/${FINETUNED_REC_MODEL_DIR_NAME} 
ENV CLS_MODEL_SUBDIR=cls/en/ch_ppocr_mobile_v2.0_cls_infer
ENV LAYOUT_MODEL_SUBDIR=layout/en/picodet_lcnet_x1_0_fgd_layout_infer

# Default to GPU 0 within the container
ENV CUDA_VISIBLE_DEVICES=0

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    python3-dev \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Update alternatives for python and pip to use python3.10
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create ALL final model directories, including the one for the fine-tuned rec model
RUN mkdir -p ${MODELS_BASE_DIR}/${DET_MODEL_SUBDIR} && \
    mkdir -p ${MODELS_BASE_DIR}/${REC_MODEL_SUBDIR} && \
    mkdir -p ${MODELS_BASE_DIR}/${CLS_MODEL_SUBDIR} && \
    mkdir -p ${MODELS_BASE_DIR}/${LAYOUT_MODEL_SUBDIR}

# Parent directories for downloaded model extraction (DET, CLS, LAYOUT)
# The parent for REC_MODEL_SUBDIR is covered by the mkdir -p above if FINETUNED_REC_MODEL_DIR_NAME is a simple name.
RUN mkdir -p ${MODELS_BASE_DIR}/det/en && \
    mkdir -p ${MODELS_BASE_DIR}/rec/en && \
    mkdir -p ${MODELS_BASE_DIR}/cls/en && \
    mkdir -p ${MODELS_BASE_DIR}/layout/en

# Download and extract BASE PaddleOCR models (Detection, Classification, Layout)
# Detection Model
RUN wget https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar -O /tmp/det_model.tar && \
    tar -xvf /tmp/det_model.tar -C ${MODELS_BASE_DIR}/det/en/ && \
    rm /tmp/det_model.tar

# Classification Model (ch_ppocr_mobile_v2.0_cls_infer)
RUN wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar -O /tmp/cls_model.tar && \
    tar -xvf /tmp/cls_model.tar -C ${MODELS_BASE_DIR}/cls/en/ && \
    rm /tmp/cls_model.tar

# Layout Model (picodet_lcnet_x1_0_fgd_layout_infer)
RUN wget https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar -O /tmp/layout_model.tar && \
    tar -xvf /tmp/layout_model.tar -C ${MODELS_BASE_DIR}/layout/en/ && \
    rm /tmp/layout_model.tar

# --- Copy your fine-tuned recognition model ---
# Source path is relative to the Docker build context (where your Dockerfile is).
# This assumes you have a directory structure like:
# ./Dockerfile
# ./local_finetuned_models/my_ppocrv4_rec_english_finetuned_runone/inference.pdiparams (and other files)
#
# The destination path is inside the image, as defined by ENV VARS.
COPY my_ppocrv4_rec_english_finetuned_runone/ ${MODELS_BASE_DIR}/${REC_MODEL_SUBDIR}/

RUN mkdir -p /opt/paddleocr_models/dicts/

COPY custom_char_dict.txt /opt/paddleocr_models/dicts/custom_char_dict.txt
# Install Python dependencies
RUN pip install --no-cache-dir -U pip setuptools wheel

# Install paddlepaddle-gpu==3.0.0 using the specific index URL for CUDA 12.6 compatible wheels
# Ensure your runtime environment matches or is compatible with CUDA 12.x.
RUN pip install --no-cache-dir paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# Copy and install the rest of your application requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copies your application source files.
COPY src/ /workspace/src/

# Copy and run the verification script
COPY docker_verify.py /workspace/docker_verify.py
RUN python /workspace/docker_verify.py

# Expose the port the app runs on
EXPOSE 5003

# Starts your model server.
CMD ["uvicorn", "src.ocr_server:app", "--host", "0.0.0.0", "--port", "5003"]