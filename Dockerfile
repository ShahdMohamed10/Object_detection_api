FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch CPU version first
RUN pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install numpy and other requirements
RUN pip install --no-cache-dir numpy==1.24.3 && pip install --no-cache-dir -r requirements.txt

# Create models directory
RUN mkdir -p /app/models

# Copy the application code
COPY . .

# Set environment variables
ENV MODEL_PATH=/app/models
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Run the application with increased timeout
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info", "--timeout-keep-alive", "120"] 