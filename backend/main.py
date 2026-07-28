from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import io
from PIL import Image

from models import cocoa_models, PredictionResponse

app = FastAPI(title="Cocoa Bean Prediction API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Cocoa Bean Prediction API is running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    model_type: str = Form(...) # "yolo" or "hybrid"
):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Resize if needed (optional, but good for consistency)
    # image = cv2.resize(image, (1280, 720)) 

    if model_type == "yolo":
        return cocoa_models.predict_yolo_only(image)
    elif model_type == "hybrid":
        return cocoa_models.predict_hybrid(image)
    else:
        raise HTTPException(status_code=400, detail="Invalid model_type. Use 'yolo' or 'hybrid'")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
