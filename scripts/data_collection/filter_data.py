"""
Downselect Crazyflie RGB-D Dataset and Visualize Camera Poses

This script performs the following tasks:

1. Loads camera extrinsics (4x4 pose matrices) from a folder of `.txt` files.
2. Loads corresponding RGB and depth images for each pose.
3. Scores each RGB image to filter out:
   - Images that are too dark (mean intensity < 20)
   - Images that are mostly black (more than 30% of pixels below a threshold)
   - Optional: low-sharpness images using Laplacian variance
4. Downselects camera views based on spatial distance:
   - Keeps only one camera if multiple cameras are too close (< 0.2 meters)
   - Prioritizes images with higher score
5. Visualizes the selected camera frustums in 3D using Open3D.
6. Saves the downselected dataset into a structured folder:
   - `down_selected/images/` → RGB images
   - `down_selected/poses/`  → Extrinsics as `.txt`
   - `down_selected/depth/`  → Depth images

"""

import numpy as np
import glob
import cv2
import open3d as o3d
import os
import shutil

# -------------------------------
# Load pose given frame ID
# -------------------------------
def load_pose_for_frame(poses_dict, frame_id):
    return np.ascontiguousarray(poses_dict[frame_id], dtype=np.float64)

# -------------------------------
# Score image (sharpness)
# -------------------------------
def score_image(image_path, darkness_threshold=10, max_darkness_fraction=0.3, min_mean_intensity=10):
    """
    Score an image based on sharpness and check for mostly black pixels using color.
    
    - threshold: pixel intensity below which a pixel is considered black
    - ratio: max fraction of black pixels allowed
    Returns np.inf if image is too dark or mostly black.
    """
    # Read in color
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_color is None:
        return np.inf

    mean_intensity = img_color.mean()

    # Count pixels that are black across all channels
    black_pixels = np.sum(np.all(img_color <= darkness_threshold, axis=2))
    total_pixels = img_color.shape[0] * img_color.shape[1]
    ratio = black_pixels / total_pixels

    # Convert to grayscale for Laplacian
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    if ratio > max_darkness_fraction or mean_intensity < min_mean_intensity:
        return np.inf 

    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    variance_score = lap.var()
    # lower score is better
    final_score = variance_score
    return final_score




# -------------------------------
# Downselect views
# -------------------------------
def downselect_views(poses_list, image_scores, min_dist=0.2):
    camera_positions = np.array([T[:3,3] for T in poses_list])
    selected = []
    sorted_idx = np.argsort(image_scores)
    for idx in sorted_idx:
        pos = camera_positions[idx]
        score = image_scores[idx]
        if score == np.inf:
            continue
        keep = True
        for j in selected:
            if np.linalg.norm(pos - camera_positions[j]) < min_dist:
                keep = False
                break
        if keep:
            selected.append(idx)
    return selected

# -------------------------------
# Camera frustum
# -------------------------------
def create_camera_frustum(scale=0.07):
    pts = np.array([
        [0,0,0],
        [-1,-1,1],
        [ 1,-1,1],
        [ 1, 1,1],
        [-1, 1,1]
    ], dtype=np.float64) * scale

    lines = np.array([
        [0,1],[0,2],[0,3],[0,4],
        [1,2],[2,3],[3,4],[4,1]
    ], dtype=np.int32)

    colors = np.tile(np.array([[1.0,0.0,0.0]], dtype=np.float64), (len(lines),1))

    fr = o3d.geometry.LineSet()
    fr.points = o3d.utility.Vector3dVector(pts)
    fr.lines = o3d.utility.Vector2iVector(lines)
    fr.colors = o3d.utility.Vector3dVector(colors)
    return fr

# -------------------------------
# Trajectory (optional)
# -------------------------------
def create_trajectory(poses):
    points = [T[:3,3] for T in poses]
    lines = [[i,i+1] for i in range(len(points)-1)]
    colors = [[0,1,0] for _ in lines]  # green
    traj = o3d.geometry.LineSet()
    traj.points = o3d.utility.Vector3dVector(points)
    traj.lines = o3d.utility.Vector2iVector(lines)
    traj.colors = o3d.utility.Vector3dVector(colors)
    return traj

# -------------------------------
# Display frustums + trajectory
# -------------------------------
def display_views(poses, frustum_scale=0.05, axis_scale=0.05):
    geoms = []

    for T in poses:
        # Camera frustum
        fr = create_camera_frustum(scale=frustum_scale)
        fr.transform(T)
        geoms.append(fr)

        # Local coordinate axis at camera
        #axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_scale)
        #axis.transform(T)  # same transform as camera
        #geoms.append(axis)

    # Optional: global coordinate frame at origin
    global_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    geoms.append(global_axis)

    # Visualize
    o3d.visualization.draw_geometries(geoms)




# -------------------------------
# Save downselected RGB, depth, and pose files
# -------------------------------
def save_downselected(filtered_rgb, filtered_poses, depth_dict, output_root="down_selected"):
    """
    Saves RGB images, poses, and depth images for downselected views.
    """
    # Create folder structure
    image_dir = os.path.join(output_root, "images")
    pose_dir  = os.path.join(output_root, "poses")
    depth_dir = os.path.join(output_root, "depth")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    for rgb_path, pose_matrix in zip(filtered_rgb, filtered_poses):
        # Frame ID
        frame_id = os.path.splitext(os.path.basename(rgb_path))[0].split('-')[-1]

        # ----------------- Save RGB -----------------
        rgb_dest = os.path.join(image_dir, os.path.basename(rgb_path))
        shutil.copyfile(rgb_path, rgb_dest)

        # ----------------- Save pose -----------------
        pose_dest = os.path.join(pose_dir, f"{frame_id}.txt")
        np.savetxt(pose_dest, pose_matrix, fmt="%.18e")

        # ----------------- Save depth -----------------
        if frame_id in depth_dict:
            depth_path = depth_dict[frame_id]
            depth_dest = os.path.join(depth_dir, os.path.basename(depth_path))
            shutil.copyfile(depth_path, depth_dest)

# -------------------------------
# Main pipeline
# -------------------------------
if __name__ == "__main__":
    # Paths
    pose_files = glob.glob("../../data/lab_data/crazyflie-poses/*.txt")
    rgb_files = glob.glob("../../data/lab_data/crazyflie-rgb-images/*.jpg")
    depth_files = glob.glob("../../data/lab_data/kinect-depth-images/*.png")  # adjust extension

    # Load poses into dictionary
    poses_dict = {}
    for f in pose_files:
        frame_id = os.path.splitext(os.path.basename(f))[0].split('-')[-1].split(".")[0]
        poses_dict[frame_id] = np.loadtxt(f)

    # Build depth mapping using same strip logic
    depth_dict = {}
    for f in depth_files:
        frame_id = os.path.splitext(os.path.basename(f))[0].split('-')[-1].split(".")[0]
        depth_dict[frame_id] = f

    # Image-driven pipeline
    poses_list = []
    rgb_list = []
    depth_list = []
    image_scores = []

    for rgb_path in rgb_files:
        frame_id = os.path.splitext(os.path.basename(rgb_path))[0].split('-')[-1].split(".")[0]
        if frame_id in poses_dict:
            poses_list.append(load_pose_for_frame(poses_dict, frame_id))
            rgb_list.append(rgb_path)
            #depth_list.append(depth_dict[frame_id])
            image_scores.append(score_image(rgb_path))

    image_scores = np.array(image_scores)
    print(f"Found {len(poses_list)} valid RGB-Depth-Pose triplets")

    # Downselect based on distance + image score
    selected_idx = downselect_views(poses_list, image_scores, min_dist=0.2)
    filtered_poses = [poses_list[i] for i in selected_idx]
    filtered_rgb = [rgb_list[i] for i in selected_idx]
    #filtered_depth = [depth_list[i] for i in selected_idx]
    filtered_scores = [image_scores[i] for i in selected_idx]

    print(f"Kept {len(filtered_poses)} / {len(poses_list)} views after downselection")
    # for score in selected_idx:
    #     img = rgb_list[score]
    #     score = image_scores[score]
    #     #print(img, score)
    for img in filtered_rgb:
        img = cv2.imread(img, cv2.IMREAD_COLOR_BGR)
        cv2.imshow("Selected", img)
        cv2.waitKey(0)
 
    # Display selected cameras
    display_views(filtered_poses)
    save_downselected(filtered_rgb, filtered_poses, depth_dict, output_root="down_selected")
