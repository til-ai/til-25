#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

VLLM_MODEL_PATH="/workspace/Qwen2.5-VL-3B-Instruct"
VLLM_HOST="0.0.0.0"
VLLM_PORT="8005"
VLLM_SERVED_MODEL_NAME="Qwen2.5-VL-3B-Instruct"

FASTAPI_HOST="0.0.0.0"
FASTAPI_PORT="5003"

# 1. Start the vLLM OpenAI API server in the background
echo "Starting vLLM OpenAI API server..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "$VLLM_MODEL_PATH" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --served-model-name "$VLLM_SERVED_MODEL_NAME" \
    --disable-log-stats & # Optional: disable periodic stat logging from vLLM if too noisy
    # Add other vLLM flags here if needed, e.g., --tensor-parallel-size

VLLM_PID=$!
echo "vLLM server started with PID $VLLM_PID on port $VLLM_PORT."

# Optional: Wait for vLLM to be ready
# A more robust check would be to curl its health endpoint if it has one,
# or wait for the port to be open. For Qwen models, loading can take time.
echo "Waiting for vLLM model to load (approx 30-90 seconds depending on model size)..."
sleep 45 # Adjust this sleep time based on your model loading observation

# Check if vLLM is still running
if ! ps -p $VLLM_PID > /dev/null; then
   echo "vLLM server failed to start. Exiting."
   exit 1
fi
echo "vLLM server presumed to be running."

# 2. Start the FastAPI/Uvicorn server for your custom OCR API in the foreground
echo "Starting FastAPI OCR server on port $FASTAPI_PORT..."
# The working directory for uvicorn needs to be /workspace for `src.ocr_server:app` to work
cd /workspace
exec uvicorn src.ocr_server:app --host "$FASTAPI_HOST" --port "$FASTAPI_PORT" --workers 1

# If uvicorn exits, the script will exit. If vLLM fails earlier, set -e handles it.