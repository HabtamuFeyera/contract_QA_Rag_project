# ==============================================================================
# Stage 1: Build React 18 Production Bundle
# ==============================================================================
FROM node:18-alpine AS frontend-builder
WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --silent

COPY frontend/ ./
# When unified, API requests use relative paths (empty string)
ENV REACT_APP_BACKEND_URL=""
RUN npm run build

# ==============================================================================
# Stage 2: Python Backend & Production Gateway
# ==============================================================================
FROM python:3.11-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application, data, and scripts
COPY backend/ ./backend/
COPY main.py .
COPY cli.py .
COPY data/ ./data/

# Copy built frontend assets for unified single-container serving
COPY --from=frontend-builder /frontend/build ./frontend/build

# Ensure runtime directories exist
RUN mkdir -p data/chroma_db data/contracts

# 7860 for Hugging Face Spaces / 8000 for standard local
EXPOSE 7860 8000

CMD ["python", "main.py"]
