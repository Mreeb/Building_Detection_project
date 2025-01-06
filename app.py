from ultralytics import YOLO
import numpy as np
import pandas as pd
import os

PREDICTION_DIR = "Detection"  # YOLO prediction directory
os.makedirs(PREDICTION_DIR, exist_ok=False)

# Scaling factor to convert pixel dimensions to real-world units (e.g., meters per pixel)
SCALING_FACTOR = 0.2  # Adjust this based on your satellite image resolution

SMALL_THRESHOLD = 100  # Example: <100 square meters
MEDIUM_THRESHOLD = 200  # Example: 100–500 square meters
MODEL = "BUILDING_DETECTION.pt"

model = YOLO(MODEL)

results = model.predict(
    source='image.jpeg',
    save=True,
    conf=0.99,
    line_width=1,
    iou= 0.1,
    project=PREDICTION_DIR
)


small_count, medium_count, big_count = 0, 0, 0

detection_info = []

for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()  
    width = x2 - x1
    height = y2 - y1
    
    area_pixels = width * height
    
    area_meters = round(area_pixels * SCALING_FACTOR**2, 2)
    
    if area_meters < SMALL_THRESHOLD:
        category = "Small"
        small_count += 1
    elif area_meters <= MEDIUM_THRESHOLD:
        category = "Medium"
        medium_count += 1
    else:
        category = "Big"
        big_count += 1

    detection = {
        "x1": round(x1),
        "y1": round(y1),
        "x2": round(x2),
        "y2": round(y2),
        "width_m": round(width * SCALING_FACTOR),
        "height_m": round(height * SCALING_FACTOR),
        "area_m": round(area_meters),
        "category": category,
    }

    detection_info.append({k: round(v) if isinstance(v, (int, float)) else v for k, v in detection.items()})

print("Detection Summary:")
print(f"Total Buildings Detected: {len(detection_info)}")
print(f"Small Buildings: {small_count}")
print(f"Medium Buildings: {medium_count}")
print(f"Big Buildings: {big_count}")

df = pd.DataFrame(detection_info)
df.to_csv("detection_summary.csv", index=False)
print("Results saved to detection_summary.csv")

