FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV MODEL_PATH=/app/models/furniture_detection.pt
ENV PORT=8080

# Run the application
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} 