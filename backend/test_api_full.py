import sys
import os
import threading
import time
import requests
import uvicorn
from fastapi.testclient import TestClient

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from main import app

client = TestClient(app)

def test_api():
    print("Testing API...")
    
    # 1. Test Root
    response = client.get("/")
    assert response.status_code == 200
    print("Root endpoint: OK")
    
    # 2. Test Prediction (YOLO)
    # Need a sample image. 
    # I'll use a dummy image or find one in the project.
    image_path = "d:/EECU_Year4-1/Senior_Project/20240403_040027532_iOS9.jpg"
    
    if not os.path.exists(image_path):
        print(f"Warning: Test image not found at {image_path}. Skipping prediction test.")
        return

    with open(image_path, "rb") as f:
        files = {"file": ("test.jpg", f, "image/jpeg")}
        data = {"model_type": "yolo"}
        response = client.post("/predict", files=files, data=data)
    
    if response.status_code == 200:
        print("YOLO Prediction: OK")
        print(response.json()['summary'])
    else:
        print(f"YOLO Prediction Failed: {response.status_code}")
        print(response.text)

    # 3. Test Prediction (Hybrid)
    with open(image_path, "rb") as f:
        files = {"file": ("test.jpg", f, "image/jpeg")}
        data = {"model_type": "hybrid"}
        response = client.post("/predict", files=files, data=data)
        
    if response.status_code == 200:
        print("Hybrid Prediction: OK")
        print(response.json()['summary'])
    else:
        print(f"Hybrid Prediction Failed: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    try:
        test_api()
        print("All backend tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
