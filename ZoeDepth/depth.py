from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import save_raw_16bit
from zoedepth.utils.misc import colorize
import numpy as np
import torch
import glob
import os

# ZoeD_K
conf = get_config("zoedepth_nk", "infer")
model_zoe_k = build_model(conf)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_k.to(DEVICE)


# Local file
from PIL import Image
# image = Image.open("./output_frames/scene_11/frame_0.png").convert("RGB")  # load

# Define source and target directories
source_dir = "/home/mpdeshmukh/RBE549-EinsteinVison/output_frames/scene_9"
target_dir_raw = "/home/mpdeshmukh/RBE549-EinsteinVison/model_outputs/zoeDepth/scene_9"

# Ensure target directories exist
os.makedirs(target_dir_raw, exist_ok=True)
# os.makedirs(target_dir_colored, exist_ok=True)

# Process each image in the source directory
for image_path in glob.glob(os.path.join(source_dir, "*.png")):
    
    # if not image_path.endswith("frame_20.png"):
    #     continue
    
    image = Image.open(image_path).convert("RGB")
    print(f"Image shape: {image.size}")
    depth = zoe.infer_pil(image)
    
    # print(f"Depth shape: {depth.shape}")
    # print(f"Depth: {depth[447, 1262]}")
    
    print(f"Processed {image_path}")

    print(depth.shape, depth.dtype, np.min(depth), np.max(depth))

    # main maal
    base_filename = os.path.basename(image_path)
    save_raw_16bit(depth, os.path.join(target_dir_raw, base_filename))


    # # Colorize output
    # colored = colorize(depth)

    # # # save colored output
    # fpath_colored = "output_colored.png"
    # Image.fromarray(colored).save(fpath_colored)
    # break