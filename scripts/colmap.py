import subprocess
from pathlib import Path
import open3d as o3d

# -------------------------------
# 1. Paths
# -------------------------------
rgb_dir = Path("./data/down_selected/images")
colmap_db = Path("colmap_database.db")
sparse_dir = Path("colmap_sparse")
dense_dir = Path("colmap_dense")
sparse_dir.mkdir(exist_ok=True)
dense_dir.mkdir(exist_ok=True)

# -------------------------------
# 2. Feature extraction
# -------------------------------
subprocess.run([
    "colmap", "feature_extractor",
    "--database_path", str(colmap_db),
    "--image_path", str(rgb_dir)
])

# -------------------------------
# 3. Feature matching (sequential)
# -------------------------------
subprocess.run([
    "colmap", "sequential_matcher",
    "--database_path", str(colmap_db)
])

# -------------------------------
# 4. Sparse reconstruction
# -------------------------------
subprocess.run([
    "colmap", "mapper",
    "--database_path", str(colmap_db),
    "--image_path", str(rgb_dir),
    "--output_path", str(sparse_dir)
])

# -------------------------------
# 5. Dense reconstruction
# -------------------------------
# 5a: Undistort images
subprocess.run([
    "colmap", "image_undistorter",
    "--image_path", str(rgb_dir),
    "--input_path", str(sparse_dir / "0"),  # first model
    "--output_path", str(dense_dir),
    "--output_type", "COLMAP",
    "--max_image_size", "2000"
])

# 5b: Patch-match stereo
subprocess.run([
    "colmap", "patch_match_stereo",
    "--workspace_path", str(dense_dir),
    "--workspace_format", "COLMAP",
    "--PatchMatchStereo.geom_consistency", "true"
])

# 5c: Stereo fusion
fused_ply = dense_dir / "fused.ply"
subprocess.run([
    "colmap", "stereo_fusion",
    "--workspace_path", str(dense_dir),
    "--workspace_format", "COLMAP",
    "--input_type", "geometric",
    "--output_path", str(fused_ply)
])

# -------------------------------
# 6. Load and visualize dense point cloud
# -------------------------------
pcd = o3d.io.read_point_cloud(str(fused_ply))
o3d.visualization.draw_geometries([pcd])
