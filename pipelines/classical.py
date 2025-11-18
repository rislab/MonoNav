import argparse
from filter_data import main
import subprocess
from pathlib import Path
import open3d as o3d
import os 


def colmap_pipeline(rgb_dir, pose_file, output_path):
    rgb_dir = Path(rgb_dir)
    pose_file = Path(pose_file)
    output_path = Path(output_path)
    colmap_db = output_path / "colmap_database.db"
    sparse_dir = output_path / "colmap_sparse"
    dense_dir = output_path / "colmap_dense"
    sparse_dir.mkdir(exist_ok=True)
    dense_dir.mkdir(exist_ok=True)

    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", str(colmap_db),
        "--image_path", str(rgb_dir)
    ])

    subprocess.run([
        "colmap", "sequential_matcher",
        "--database_path", str(colmap_db)
    ])

    subprocess.run([
        "colmap", "mapper",
        "--database_path", str(colmap_db),
        "--image_path", str(rgb_dir),
        "--output_path", str(sparse_dir)
    ])

    subprocess.run([
        "colmap", "image_undistorter",
        "--image_path", str(rgb_dir),
        "--input_path", str(sparse_dir / "0"),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",
        "--max_image_size", "2000"
    ])

    subprocess.run([
        "colmap", "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true"
    ])

    fused_ply = dense_dir / "fused.ply"
    subprocess.run([
        "colmap", "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(fused_ply)
    ])

    pcd = o3d.io.read_point_cloud(str(fused_ply))
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run COLMAP pipeline on images and poses")
    parser.add_argument("--data_path", required=True, help="Data directory containing your experiment")
    parser.add_argument("--output_path", default="colmap", help="Output directory")
    args = parser.parse_args()

    data_path = args.data_path
    output_path = os.path.join(data_path, args.output_path)
    rgb_path = os.path.join(data_path, "crazyflie-rgb-images")
    pose_path= os.path.join(data_path, "crazyflie-poses")
    
    # Filtering + Undistorition 
    main(pose_path, rgb_path, output_name=output_path)

    # Run Colmap
    down_selected_rgb_path = os.path.join(output_path, "images")
    down_sampled_poses_path = os.path.join(output_path, "poses")
    colmap_pipeline(down_selected_rgb_path, down_sampled_poses_path, output_path)
