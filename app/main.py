from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from .model import FurnitureDetector
import os
import logging
import sys

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Furniture Detection API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
model_path = os.getenv('MODEL_PATH', '/app/models/furniture_detection.pt')
logger.info(f"Initializing model with path: {model_path}")
try:
    model = FurnitureDetector(model_path)
    logger.info("Model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing model: {str(e)}")
    model = None

@app.post("/detect")
async def detect_furniture(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not initialized", "status": "error"}
    
    try:
        contents = await file.read()
        results = await model.predict(contents)
        return results
    except Exception as e:
        logger.error(f"Error in detect_furniture: {str(e)}")
        return {"error": str(e), "status": "error"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None} 