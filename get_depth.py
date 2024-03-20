from ZoeDepth.zoedepth.models.builder import build_model
from ZoeDepth.zoedepth.utils.config import get_config
from ZoeDepth.zoedepth.utils.misc import save_raw_16bit, colorize
import cv2  # Import OpenCV
from PIL import Image
import torch
import os

scene = "scene_11"

# Load the model configuration and model
conf = get_config("zoedepth", "infer", config_version="kitti")
model_zoe_k = build_model(conf)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_k.to(DEVICE)

# Define the output directory paths for raw and colorized images
raw_output_dir = f"./model_outputs/zoeDepth/{scene}/raw"
colorized_output_dir = f"./model_outputs/zoeDepth/{scene}/colored"

# Ensure output directories exist
os.makedirs(raw_output_dir, exist_ok=True)
os.makedirs(colorized_output_dir, exist_ok=True)

# Function to process and save a single frame
def process_and_save_frame(frame, frame_index):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
    depth = zoe.infer_pil(image)
    
    # Save raw depth map
    raw_fpath = f"{raw_output_dir}/frame_{frame_index}.png"
    save_raw_16bit(depth, raw_fpath)
    
    # Save colorized depth map
    colored = colorize(depth)
    colored_fpath = f"{colorized_output_dir}/frame_{frame_index}_colored.png"
    Image.fromarray(colored).save(colored_fpath)

# Load video
video_path = "/video.mp4"
cap = cv2.VideoCapture(video_path)

frame_interval = 10  # Process every 10th frame
frame_index = 0

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break  # Break if no more frames are available
    
    if frame_index % frame_interval == 0:
        process_and_save_frame(frame, frame_index)
    
    frame_index += 1

cap.release()
