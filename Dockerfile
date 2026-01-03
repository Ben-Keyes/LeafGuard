FROM node:20-alpine AS frontend_builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
ENV CI=false
RUN npm run build

# Base image
FROM python:3.10-slim

WORKDIR /app

# System dependencies for PyTorch an Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install backend Python dependencies
COPY backend/requirements.txt .
RUN pip install -r requirements.txt

# Install CPU-only PyTorch
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Copy backend and ML model
COPY backend/ ./backend
WORKDIR /app/backend

# Copy React build directly into backend/static/frontend
#COPY frontend/build ./static/frontend
COPY --from=frontend_builder /frontend/build ./static/frontend

# Exposing port for FastAPI
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]