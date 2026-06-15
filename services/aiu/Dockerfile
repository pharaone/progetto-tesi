FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first so sentence-transformers doesn't pull CUDA wheels
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu

# Copy requirements first for layer caching
COPY services/aiu/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared modules
COPY shared/ /app/shared/
COPY rag/ /app/rag/

# Copy service code
COPY services/aiu/main.py /app/main.py

# Create data directory
RUN mkdir -p /data/chromadb

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8005

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]
