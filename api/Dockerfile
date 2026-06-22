# ── Dockerfile untuk FastAPI ──────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Pre-download model IndoBERT saat build (agar startup cepat)
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
model_name = 'w11wo/indonesian-roberta-base-sentiment-classifier'; \
AutoTokenizer.from_pretrained(model_name); \
AutoModelForSequenceClassification.from_pretrained(model_name); \
print('Model downloaded successfully')"

# Copy source code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
