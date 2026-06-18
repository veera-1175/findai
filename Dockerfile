FROM python:3.11-slim-bookworm

# OpenCV, audio decode (ffmpeg for MP3/M4A)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Hugging Face model cache (ephemeral on free Render — re-downloads on cold start)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RENDER=true \
    CLOUD=true \
    DEBUG=false \
    PRELOAD_ML=false \
    HOST=0.0.0.0 \
    DATA_DIR=/tmp/findai \
    HF_HOME=/tmp/huggingface \
    TRANSFORMERS_CACHE=/tmp/huggingface \
    TORCH_HOME=/tmp/torch

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY run.py .

EXPOSE 8000

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 120
