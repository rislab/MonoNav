import json
import numpy as np
import open3d as o3d
from pathlib import Path

# Load camera intrinsics
with open("utils/calibration/intrinsics.json", "r") as f:
    intr = json.load(f)

K = np.array(intr["CameraMatrix"])
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]
width, height = intr["ROI"][2], intr["ROI"][3]

intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# Directory setup

rgb_dir = Path("./data/down_selected/images")
depth_dir = Path("./data/down_selected/depth")
poses_dir = Path("./data/down_selected/poses")

# Sort files to ensure correspondence
rgb_files = sorted(rgb_dir.glob("*.jpg"))
depth_files = sorted(depth_dir.glob("*.jpg"))
pose_files = sorted(poses_dir.glob("*.txt"))



num_samples = len(rgb_files) #min(len(rgb_files), len(depth_files), len(pose_files))

rgb_files = rgb_files[:num_samples]
depth_files = depth_files[:num_samples]
pose_files = pose_files[:num_samples]

# Combine point clouds

pcd_combined = o3d.geometry.PointCloud()

for rgb_path, depth_path, pose_path in zip(rgb_files, depth_files, pose_files):
    color = o3d.io.read_image(str(rgb_path))
    depth = o3d.io.read_image(str(depth_path))

    # Adjust depth_scale if needed (millimeters to meters)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth,
        depth_scale=1,
        depth_trunc=4.0,
        convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    
    # Flip the point cloud to align with the correct coordinate system
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])
    
    # Load and apply camera pose (4x4 matrix)
    pose = np.loadtxt(str(pose_path))
    pcd.transform(pose)

    pcd_combined += pcd

# Filter noise
#pcd_combined, ind = pcd_combined.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
#pcd_combined = pcd_combined.voxel_down_sample(voxel_size=0.01)
#pcd_combined.estimate_normals()

# Visualize and save
o3d.visualization.draw_geometries([pcd_combined])
