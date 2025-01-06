import os
from PIL import Image

source_folder = "ORIGNAL_DATA"
destination_folder = "90_Degree_Rotation"

os.makedirs(destination_folder, exist_ok=True)

for filename in os.listdir(source_folder):
    source_path = os.path.join(source_folder, filename)
    
    if os.path.isfile(source_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
        try:
            with Image.open(source_path) as img:
                rotated_img = img.rotate(90, expand=True)
                
                destination_path = os.path.join(destination_folder, filename)
                rotated_img.save(destination_path)
                
                print(f"Processed: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("Image rotation complete!")
