# YOLOv8 Furniture Detection API

This project deploys a YOLOv8 model on Google Cloud for furniture detection in AR applications.

## Features
- Object detection for furniture (chairs, tables, sofas, etc.)
- REST API endpoints for image processing
- Deployment on Google Cloud (Vertex AI)
- Containerized with Docker

## Project Structure
```
object_detection/
├── app/
│   ├── main.py           # FastAPI application
│   ├── model.py          # YOLOv8 model wrapper
│   └── utils.py          # Utility functions
├── config/
│   └── config.yaml       # Configuration file
├── Dockerfile            # For containerization
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
└── scripts/
    └── deploy_vertex.py  # Deployment script
```

## Setup
1. Clone the repository
```bash
git clone <your-repo-url>
cd object_detection
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up environment variables
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Deployment
1. Build Docker image
```bash
docker build -t furniture-detection .
```

2. Deploy to Google Cloud
```bash
python scripts/deploy_vertex.py
```

## API Endpoints
- POST /detect - Detect furniture in images
- GET /health - Health check endpoint

## Environment Variables
- `MODEL_PATH`: Path to YOLOv8 model file
- `GOOGLE_CLOUD_PROJECT`: Google Cloud project ID
- `PORT`: API port (default: 8080) 