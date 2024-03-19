from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import save_raw_16bit
from zoedepth.utils.misc import colorize
import numpy as np
import torch

# ZoeD_K
conf = get_config("zoedepth", "infer", config_version="kitti")
model_zoe_k = build_model(conf)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_k.to(DEVICE)


# Local file
from PIL import Image
image = Image.open("./output_frames/scene_11/frame_0.png").convert("RGB")  # load

depth = zoe.infer_pil(image)

print(depth.shape, depth.dtype, np.min(depth), np.max(depth))

# main maal
fpath = "ZoeDepth/output.png"
save_raw_16bit(depth, fpath)

# Colorize output
colored = colorize(depth)

# save colored output
fpath_colored = "ZoeDepth/output_colored.png"
Image.fromarray(colored).save(fpath_colored)