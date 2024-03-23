# Front Camera intrinsics
# fx = 1594.7
# fy = 1607.7
# cx = 655.2961
# cy = 414.3627
# Image size: 1280 x 960
# 1 00
# 0-10
# 0 0 -1
import json
import numpy as np
from PIL import Image
import cv2

# Camera intrinsic parameters
fx = 1594.7
fy = 1607.7
cx = 655.2961
cy = 414.3627

# Function to convert from pixel coordinates to real-world coordinates
def pixel_to_world(u, v, z):
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return x, y, z

# Load results.json
with open("/home/mpdeshmukh/RBE549-EinsteinVison/model_outputs/yolov9/scene_1/results.json", "r") as file:
    results = json.load(file)

# Process each frame's objects and update positions
def process_frames(frames_data):
    final_data = []
    for frame_data in frames_data:
        frame_number = frame_data['frame']
        depth_image_path = f"/home/mpdeshmukh/RBE549-EinsteinVison/model_outputs/zoeDepth/scene_1/frame_{frame_number}.png"
        
        # Open and load the depth image
        depth_image = np.array(Image.open(depth_image_path))
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
        
        print(f"Processing frame {frame_number} with depth image shape: {depth_image.shape}")
        print(f"Depth min: {np.min(depth_image)}, max: {np.max(depth_image)}")
        # if not depth_image_path.endswith("frame_20.png"):
        #     continue
        
        # while True:
        #     ...

        # Update objects with real-world coordinates
        objects = frame_data.get("objects")
        if objects is None:
            objects = []

        # Update objects with real-world coordinates
        updated_objects = []
        for obj in objects:
            u = obj["bbx"]["x"]
            v = obj["bbx"]["y"]
            # Ensure z is a native Python type (convert numpy int to Python float)
            z = float(depth_image[int(v), int(u)])/1000  # Convert to float for JSON serialization
            
            # Convert to real-world coordinates
            x_world, y_world, z_world = pixel_to_world(u, v, z)
            
            updated_obj = {
                "type": obj["type"],
                "position": {"x": x_world, "y": y_world, "z": z_world},
                "rotation": obj.get("rotation", {"x": 0, "y": 0, "z": 0}),
                "scale": obj.get("scale", {"x": 1, "y": 1, "z": 1})
            }
            updated_objects.append(updated_obj)

        final_data.append({"frame": frame_number, "objects": updated_objects})
    
    return final_data

# Process all frames and write to new JSON file
final_json_data = process_frames(results)
with open("/home/mpdeshmukh/RBE549-EinsteinVison/model_outputs/spawn_adjusted.json", "w") as outfile:
    json.dump(final_json_data, outfile, indent=4)
