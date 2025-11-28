import os
import cv2
import numpy as np

DATA_ROOT = "data"

IMAGE_FOLDER = "crazyflie-rgb-images"
CF_POSE_FOLDER = "crazyflie-poses"
TF_POSE_FOLDER = "crazyflie-tf-poses"

BAD_IMAGE_THRESHOLD = 40
MAX_POSE_JUMP = 2  # meters

def is_bad_image(image_path, threshold=BAD_IMAGE_THRESHOLD):
    img = cv2.imread(str(image_path))
    if img is None:
        return True  # treat unreadable image as black
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    return mean_val < threshold


def load_pose(pose_path):
    return np.loadtxt(pose_path)

def is_bad_pose(pose1, pose2, max_jump=MAX_POSE_JUMP):
    pos1 = pose1[:3, 3]
    pos2 = pose2[:3, 3]
    dist = np.linalg.norm(pos2 - pos1)
    return dist > max_jump

def filter_dataset(dataset_path):
    img_dir = os.path.join(dataset_path, IMAGE_FOLDER)
    cf_pose_dir = os.path.join(dataset_path, CF_POSE_FOLDER)
    tf_pose_dir = os.path.join(dataset_path, TF_POSE_FOLDER)
    ignore_tf = True if str(dataset_path).find("flow") != -1 else False

    img_files = sorted(os.listdir(img_dir))
    cf_pose_files = sorted(os.listdir(cf_pose_dir))
    if not ignore_tf:
        tf_pose_files = sorted(os.listdir(tf_pose_dir)) 
    else:
        tf_pose_files = cf_pose_files

    last_cf_pose = None
    last_tf_pose = None
    counter = 0
    for img_file, cf_pose_file, tf_pose_file in zip(img_files, cf_pose_files, tf_pose_files):
        
        img_path = os.path.join(img_dir, img_file)
        cf_pose_path = os.path.join(cf_pose_dir, cf_pose_file)
        tf_pose_path = os.path.join(tf_pose_dir, tf_pose_file)

        remove = False

        # Check image
        if is_bad_image(img_path):
            print(f"[{dataset_path}] Removing bad image: {img_file}")
            remove = True

        # Check cf3 pose jump
        cf_pose = load_pose(cf_pose_path)
        if last_cf_pose is not None and is_bad_pose(last_cf_pose, cf_pose):
            print(f"[{dataset_path}] Removing bad /cf3/pose: {cf_pose_file}")
            remove = True

        # Check tf pose jump
        if not ignore_tf:
            tf_pose = load_pose(tf_pose_path)
            if last_tf_pose is not None and is_bad_pose(last_tf_pose, tf_pose):
                print(f"[{dataset_path}] Removing bad /tf pose: {tf_pose_file}")
                remove = True
        else:
            tf_pose = None

        if remove:
           os.remove(img_path)
           os.remove(cf_pose_path)
           if not ignore_tf:
            os.remove(tf_pose_path)
           #print("YES REMOVING", img_file)
           #counter += 1 
        else:
            last_cf_pose = cf_pose
            last_tf_pose = tf_pose

    print("removed:", counter)
if __name__ == "__main__":
    datasets = [os.path.join(DATA_ROOT, d) for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d)) and d != "archive" and d != "demo_hallway"]

    for dataset in datasets:
        print(f"Filtering dataset: {dataset}")
        filter_dataset(dataset)

    print("Done filtering all datasets.")
