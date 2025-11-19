import argparse
from filter_data import main, get_intrinsics
import subprocess
from pathlib import Path
import open3d as o3d
import os 
from PIL import Image
import numpy as np 
import shutil

# -------------------------------
# COLMAP pipeline without using external intrinsics & poses
# -------------------------------
def colmap_pipeline_auto(rgb_dir, output_path, quality="low"):
    """
    Run COLMAP automatic reconstruction (COLMAP estimates intrinsics & extrinsics).
    """

    rgb_dir = Path(rgb_dir)
    output_path = Path(output_path)

    # Make sure workspace exists
    output_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "colmap", "automatic_reconstructor",
        "--workspace_path", str(output_path),
        "--image_path", str(rgb_dir),
        "--quality", quality,
        "--use_gpu", "1"
    ]

    print("\nRunning COLMAP automatic reconstruction:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    # Load dense fused point cloud (COLMAP standard output path)
    fused_ply = output_path / "dense/0" / "fused.ply"

    if fused_ply.exists():
        print("Loading fused point cloud:", fused_ply)
        pcd = o3d.io.read_point_cloud(str(fused_ply))
        o3d.visualization.draw_geometries([pcd])
    else:
        print("WARNING: fused.ply not found. Dense reconstruction may have failed.")


# -------------------------------
# Helper functions
# -------------------------------
def rotation_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion (qw, qx, qy, qz)"""
    qw = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2
    qx = (R[2,1] - R[1,2]) / (4*qw)
    qy = (R[0,2] - R[2,0]) / (4*qw)
    qz = (R[1,0] - R[0,1]) / (4*qw)
    return qw, qx, qy, qz

def load_pose(pose_file):
    """Load 4x4 homogeneous pose matrix from TXT"""
    M = np.loadtxt(pose_file)
    R = M[:3, :3]
    t = M[:3, 3]
    return R, t

# -------------------------------
# Pipeline: Known poses + intrinsics
# -------------------------------
def colmap_pipeline_external(rgb_dir, pose_dir, output_path):
    rgb_dir = Path(rgb_dir)
    pose_dir = Path(pose_dir)
    output_path = Path(output_path)

    output_path.mkdir(parents=True, exist_ok=True)
    images_path = output_path / "images"
    images_path.mkdir(exist_ok=True)

    # copy images over to workspace
    for i, img_file in enumerate(rgb_dir.glob("*")):
        shutil.copy(img_file, images_path / img_file.name)


    # -------------------------------
    # 1. Get intrinsics
    # -------------------------------
    mtx, dist, _ = get_intrinsics()
    fx, fy = mtx[0,0], mtx[1,1]
    cx, cy = mtx[0,2], mtx[1,2]
    k = dist[0] if len(dist) > 0 else 0.0

    # -------------------------------
    # 2. Create manual sparse model
    # -------------------------------
    sparse_dir = output_path / "manual_sparse"
    sparse_dir.mkdir(exist_ok=True)
    cameras_txt = sparse_dir / "cameras.txt"
    images_txt = sparse_dir / "images.txt"
    points3D_txt = sparse_dir / "points3D.txt"

    # Cameras.txt
    first_image = next(rgb_dir.glob("*"))
    w, h = Image.open(first_image).size
    with cameras_txt.open("w") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 SIMPLE_RADIAL {w} {h} {fx} {cx} {cy} {k}\n")

    # Images.txt
    image_files = sorted(rgb_dir.glob("*"))
    pose_files = sorted(pose_dir.glob("*.txt"))
    with images_txt.open("w") as f:
        f.write("# IMAGE_ID, QW QX QY QZ, TX TY TZ, CAMERA_ID, NAME\n")
        for idx, (img_file, pose_file) in enumerate(zip(image_files, pose_files), 1):
            R, t = load_pose(pose_file)
            qw, qx, qy, qz = rotation_to_quaternion(R)
            # First line: pose
            f.write(f"{idx} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 {img_file.name}\n")
            # Second line: empty keypoints
            f.write("\n")
    # Empty points3D.txt
    points3D_txt.write_text("# Empty: will be filled after triangulation\n")

    # -------------------------------
    # 3. COLMAP feature extraction
    # -------------------------------
    db_path = output_path / "database.db"
    db_path = db_path.resolve()

    subprocess.run([
    "colmap", "feature_extractor",
    "--database_path", str(db_path),
    "--image_path", str(images_path),  # absolute path
    "--ImageReader.camera_model", "SIMPLE_RADIAL",
    "--ImageReader.camera_params", f"{fx},{cx},{cy},{k}",
    "--ImageReader.single_camera", "1",
    "--FeatureExtraction.use_gpu", "1"
    ], check=True)

    # -------------------------------
    # 4. Feature matching
    # -------------------------------
    subprocess.run([
    "colmap", "exhaustive_matcher",
    "--database_path", str(db_path),
    "--FeatureMatching.use_gpu", "1"
    ], check=True)

    # -------------------------------
    # 5. Triangulation using known poses
    # -------------------------------
    triangulated_dir = output_path / "sparse_triangulated"
    triangulated_dir.mkdir(exist_ok=True)
    subprocess.run([
        "colmap", "point_triangulator",
        "--database_path", str(db_path),
        "--image_path", str(images_path),
        "--input_path", str(sparse_dir),
        "--output_path", str(triangulated_dir),
        "--Mapper.ba_refine_focal_length", "0",       # keep focal fixed
        "--Mapper.ba_refine_principal_point", "0",    # keep principal fixed
        "--Mapper.ba_refine_extra_params", "0",       # keep distortion fixed
    ], check=True)





    print("Sparse reconstruction completed using known poses and intrinsics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run COLMAP pipeline on images and poses")
    parser.add_argument("--data_path", required=True, help="Data directory containing your experiment")
    parser.add_argument("--output_path", default="colmap-pipeline", help="Output directory")
    args = parser.parse_args()

    data_path = args.data_path
    output_path = os.path.join(data_path, args.output_path)
    rgb_path = os.path.join(data_path, "crazyflie-rgb-images")
    pose_path= os.path.join(data_path, "crazyflie-poses")
    
    # Filtering + Undistorition 
    #down_selected_rgb_path, down_sampled_poses_path = main(pose_path, rgb_path, output_name=output_path)

    # Run Colmap 
    #colmap_pipeline_auto(down_selected_rgb_path, output_path)
    colmap_pipeline_external(rgb_path, pose_path, output_path)
