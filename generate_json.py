import json
import os

frames_dir = "path/to/frames"
output_json = []

for frame_file in os.listdir(frames_dir):
    lane_data = read_json(f"{frames_dir}/{frame_file}_lane.json")
    object_data = read_json(f"{frames_dir}/{frame_file}_object.json")
    depth_data = read_json(f"{frames_dir}/{frame_file}_depth.json")
    
    frame_data = aggregate_data(lane_data, object_data, depth_data)
    output_json.append(frame_data)

with open("final_for_blender.json", "w") as f:
    json.dump(output_json, f)

def read_json(file_path):
    with open(file_path) as f:
        return json.load(f)

def aggregate_data(lane_data, object_data, depth_data):
    # Implement the aggregation logic here
    # Calculate real-world coordinates and prepare the data structure
    return aggregated_data
