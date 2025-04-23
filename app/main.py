from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from .model import FurnitureDetector
import os
from dotenv import load_dotenv

load_dotenv()

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
model = FurnitureDetector(os.getenv('MODEL_PATH'))

@app.post("/detect")
async def detect_furniture(file: UploadFile = File(...)):
    contents = await file.read()
    results = await model.predict(contents)
    return results

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 