import time
import os
import sys
import torch
import cv2

# Add ZoeDepth to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
sys.path.insert(0, os.path.join(parent_dir, "ZoeDepth"))
sys.path.insert(0, os.path.join(parent_dir, "Depth-Anything-V2"))

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

import open3d as o3d
from PIL import Image
import numpy as np
import matplotlib

from utils.utils import compute_depth, load_config, get_calibration_values, transform_image

""""
This script runs a depth estimation model on a directory of RGB images and saves the depth images.
"""

cmap = matplotlib.colormaps.get_cmap('Spectral')

# -------------------------------
# Get Intrinsics
# -------------------------------
def get_intrinsics():
    config = load_config(os.path.join(parent_dir, "config.yml"))

    # Load the calibration values
    camera_calibration_path = config["camera_calibration_path"]
    mtx, dist = get_calibration_values(camera_calibration_path)
    # Kinect intrinsic matrix
    kinect = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    return mtx, dist, kinect


# -------------------------------
# Load ZoeDepth
# -------------------------------
def load_zoedepth(config):
    conf = get_config("zoedepth", config["zoedepth_mode"]) # NOTE: "eval" runs slightly slower, but is stated to be more metrically accurate
    model_zoe = build_model(conf)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device is: ", DEVICE)
    zoe = model_zoe.to(DEVICE)
    return zoe

# -------------------------------
# Load DepthAny
# -------------------------------
def load_depthany(max_depth = 20):
    checkpoint = 'Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model_config = {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    depth_anything = DepthAnythingV2(**{**model_config, 'max_depth': max_depth})
    depth_anything.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    return depth_anything


def convert_depth(model, dir, model_type = "zoe"):
    image_dir = os.path.join(dir, "images")
    depth_dir = os.path.join(dir, "depth", model_type)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    # Figure out how many images are in folder by counting .jpg files
    rgb_files = [os.path.join(image_dir, name) for name in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, name)) and name.endswith(".jpg")][:-1]

    start_time = time.time()
    frame_number = 0
    end_frame = len(rgb_files)
    for rgb_file in rgb_files:
        print(f"Applying {model_type} to:  %d/%d"%(frame_number+1,end_frame))

        # Compute depth
        if model_type == "zoe":
            # Read in image with Pillow and convert to RGB
            rgb = Image.open(rgb_file).convert("RGB")
            depth_numpy, depth_colormap = compute_depth(rgb, model)
        elif model_type == "depthany":
            # Read in image with Pillow and convert to RGB
            rgb = cv2.imread(rgb_file)
            depth_numpy = model.infer_image(rgb, 518)
            
            depth = (depth_numpy - depth_numpy.min()) / (depth_numpy.max() - depth_numpy.min()) * 255.0
            depth = depth.astype(np.uint8)
            depth_colormap = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        # Save images
        depth_path = os.path.join(depth_dir, os.path.basename(rgb_file).replace("rgb", "depth"))[:-4]
        cv2.imwrite(depth_path + ".jpg", depth_colormap)
        np.save(depth_path + ".npy", depth_numpy) # saved in meters

        frame_number += 1

    print("Time to compute depth for %d images: %f"%(end_frame, time.time()-start_time))

def main(dir, model_type = "zoe"):
    if model_type == "zoe":
        config = load_config("config.yml")
        model = load_zoedepth(config)
    elif model_type == "depthany":
        model = load_depthany(max_depth=20)
    
    convert_depth(model = model, dir = dir, model_type = model_type)


if __name__ == "__main__":    
    main('/home/nicholas/MonoNav/down_selected/', model_type="zoe")
    main('/home/nicholas/MonoNav/down_selected/', model_type="depthany")