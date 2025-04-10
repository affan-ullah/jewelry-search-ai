# main.py
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
import torch
import clip
from PIL import Image
import io
import numpy as np
import os
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS with environment variables
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
preprocess = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Lazy-loading function for the model
def get_model():
    global model, preprocess
    if model is None:
        logger.info(f"Loading CLIP model on {device}...")
        model, preprocess = clip.load("ViT-B/32", device=device)
        logger.info("Model loaded successfully")
    return model, preprocess

@app.get("/")
def read_root():
    return {
        "message": "Jewelry Visual Search AI Service", 
        "status": "Running",
        "device": device
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "device": device}

@app.post("/generate-embedding")
async def generate_embedding(file: UploadFile = File(...)):
    try:
        # Load model if not already loaded
        model, preprocess = get_model()
        
        # Read the image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        logger.info(f"Processing image: {file.filename}, size: {image.size}")
        
        # Preprocess the image and generate embedding
        with torch.no_grad():
            image_input = preprocess(image).unsqueeze(0).to(device)
            image_features = model.encode_image(image_input)
            
            # Normalize the features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Convert to list for JSON serialization
            embedding = image_features.cpu().numpy().tolist()[0]
        
        logger.info(f"Embedding generated successfully, length: {len(embedding)}")
        return {"embedding": embedding}
    
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return {"error": str(e)}, 500

# Load model on startup only if environment variable is set
@app.on_event("startup")
async def startup_event():
    if os.environ.get("PRELOAD_MODEL", "false").lower() == "true":
        get_model()
        logger.info("Model preloaded on startup")
    else:
        logger.info("Model will be loaded on first request")