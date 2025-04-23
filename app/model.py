from ultralytics import YOLO
from typing import List, Dict
import numpy as np
from PIL import Image
import io
import os

class FurnitureDetector:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}, downloading pre-trained model...")
            self.model = YOLO('yolov8n.pt')  # Use a pre-trained model
        else:
            self.model = YOLO(model_path)
        self.furniture_classes = ['chair', 'table', 'sofa', 'bed', 'cabinet']

    async def predict(self, image_bytes: bytes) -> Dict:
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Run inference
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
                
            return {
                'detections': detections,
                'image_size': image.size
            }
        except Exception as e:
            return {
                'error': str(e),
                'status': 'error'
            } 