from ultralytics import YOLO
from typing import List, Dict
import numpy as np
from PIL import Image
import io
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FurnitureDetector:
    def __init__(self, model_path: str):
        try:
            if not os.path.exists(model_path):
                logger.warning(f"Model not found at {model_path}, downloading pre-trained model...")
                self.model = YOLO('yolov8n.pt')  # Use a pre-trained model
                logger.info("Successfully loaded pre-trained model")
            else:
                self.model = YOLO(model_path)
                logger.info(f"Successfully loaded model from {model_path}")
            self.furniture_classes = ['chair', 'table', 'sofa', 'bed', 'cabinet']
            logger.info(f"Initialized FurnitureDetector with classes: {self.furniture_classes}")
        except Exception as e:
            logger.error(f"Error initializing FurnitureDetector: {str(e)}")
            raise

    async def predict(self, image_bytes: bytes) -> Dict:
        try:
            logger.info("Starting prediction process")
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            logger.info(f"Successfully loaded image with size: {image.size}")
            
            # Run inference
            logger.info("Running YOLO inference")
            results = self.model(image, conf=0.25)[0]
            
            # Process results
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = results.names[class_id]
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'class': class_name,
                    'confidence': confidence
                })
            
            logger.info(f"Found {len(detections)} objects in the image")
            return {
                'detections': detections,
                'image_size': image.size,
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return {
                'error': str(e),
                'status': 'error'
            } 