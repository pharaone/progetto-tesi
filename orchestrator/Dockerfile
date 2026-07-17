FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY orchestrator/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared modules
COPY shared/ /app/shared/
COPY rag/ /app/rag/
COPY connectors/ /app/connectors/

# Copy orchestrator code
COPY orchestrator/__init__.py /app/orchestrator/__init__.py
COPY orchestrator/graph.py /app/orchestrator/graph.py
COPY orchestrator/db.py /app/orchestrator/db.py
COPY orchestrator/main.py /app/orchestrator/main.py

# Create data directory
RUN mkdir -p /data/chromadb /data/app

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "orchestrator.main:app", "--host", "0.0.0.0", "--port", "8000"]
