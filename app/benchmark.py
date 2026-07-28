import sys
import os
import cv2
import time

# Add path so we can import app_cpu
sys.path.append(r"d:\Chula\Senior_Project\app")
import app_cpu

def run_benchmark():
    img_path = r"d:\Chula\Senior_Project\app\static\uploads\20240403_040027532_iOS.png"
    # Read and resize exactly like the web server does
    frame = cv2.imread(img_path)
    if frame is None:
        print("Failed to load image")
        return
    frame = cv2.resize(frame, (1280, 720))
    
    # Warm up models
    print("===== Warming up models =====")
    app_cpu.process_ai_background(frame, "combo")
    
    # Run test
    print("\n===== Running actual test (Baseline) =====")
    runs = 3
    for i in range(runs):
        # We need to ensure is_processing is False before calling
        app_cpu.is_processing = False
        app_cpu.process_ai_background(frame, "combo")
        prof = app_cpu.latest_detection["profiling"]
        print(f"Run {i+1} Profiling:")
        for k, v in prof.items():
            print(f"  {k}: {v} sec")
        print("-" * 30)

if __name__ == "__main__":
    run_benchmark()
