import os
from ultralytics import YOLO
import pandas as pd
import gradio as gr
from PIL import Image

PREDICTION_DIR = "Detection"  # YOLO prediction directory
os.makedirs(PREDICTION_DIR, exist_ok=False)

SCALING_FACTOR = 0.2  # Real-world unit conversion factor
SMALL_THRESHOLD = 100  # Small building threshold in square meters
MEDIUM_THRESHOLD = 200  # Medium building threshold in square meters
MODEL_PATH = "BUILDING_DETECTION.pt"
PREDICTION_DIR = "Detection"  # YOLO prediction directory

# Load the YOLO model
model = YOLO(MODEL_PATH)

def predict_and_display(image):
    """Process the uploaded image, run YOLO model, and return results."""
    results = model.predict(source=image, save=True, conf=0.99,iou=0.1, line_width=1, project=PREDICTION_DIR)

    latest_dir = max([os.path.join(PREDICTION_DIR, d) for d in os.listdir(PREDICTION_DIR)], key=os.path.getmtime)
    predicted_image_path = next(
        (os.path.join(latest_dir, f) for f in os.listdir(latest_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))), 
        None
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

        detection_info.append({
            "x1": round(x1),
            "y1": round(y1),
            "x2": round(x2),
            "y2": round(y2),
            "width_m": round(width * SCALING_FACTOR),
            "height_m": round(height * SCALING_FACTOR),
            "area_m": round(area_meters),
            "category": category,
        })

    detection_summary = pd.DataFrame(detection_info)
    csv_path = os.path.join("detection_summary.csv")
    detection_summary.to_csv(csv_path, index=False)

    summary_text = (
        f"Total Buildings Detected: {len(detection_info)}\n"
        f"Small Buildings: {small_count}\n"
        f"Medium Buildings: {medium_count}\n"
        f"Big Buildings: {big_count}"
    )

    
    map50_Score_path = "Logs/map50_Score.png"
    map50_95_Score_path = "Logs/map50-95_Score.png"
    precision_Score_path = "Logs/Precision_Score.png"
    recall_Score_path = "Logs/Recall_Score.png"

    return (
        Image.open(predicted_image_path),
        summary_text,
        detection_summary,
        Image.open(map50_Score_path),
        Image.open(map50_95_Score_path),
        Image.open(precision_Score_path),
        Image.open(recall_Score_path),
    )

with gr.Blocks() as app:
    gr.Markdown("# Building Detection App")
    gr.Markdown("Upload an image to detect buildings and classify them by size.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="Upload Image")

            detect_button = gr.Button("Detect")

        with gr.Column():
            output_image = gr.Image(label="Predicted Image")

    summary_output = gr.Text(label="Detection Summary")
    csv_output = gr.DataFrame(label="Detection Details")

    with gr.Row():
        with gr.Column():
            map50_Score = gr.Image(label="Map50 Score")
            map50_95_Score = gr.Image(label="Map50-95 Score")

        with gr.Column():
            precision_Score = gr.Image(label="Precision Score")
            recall_Score = gr.Image(label="Recall Score")
            
    # log_image1 = gr.Image(label="Logs1")
    # log_image2 = gr.Image(label="Logs2")

    # I want to add to more images just below this 
    # are at location Logs/logs1 and Logs/logs2

    

    # Button action to process image
    detect_button.click(
        predict_and_display, 
        inputs=image_input, 
        outputs=[
            output_image,
            summary_output,
            csv_output,
            map50_Score,
            map50_95_Score,
            precision_Score,
            recall_Score
        ]
    )

app.launch()