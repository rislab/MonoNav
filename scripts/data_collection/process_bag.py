import rosbag
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import time
from scipy.spatial.transform import Rotation

# -------------------------------
# CONFIG / SAVE PATH
# -------------------------------
bag_path = 'cf3_data.bag'

save_dir = 'data/trial-from-bag-' + time.strftime("%Y-%m-%d-%H-%M-%S")
crazyflie_img_dir = os.path.join(save_dir, "crazyflie-rgb-images")
crazyflie_pose_dir = os.path.join(save_dir, "crazyflie-poses")

os.makedirs(crazyflie_img_dir, exist_ok=True)
os.makedirs(crazyflie_pose_dir, exist_ok=True)

print("Saving files to:", save_dir)

# -------------------------------
# LOAD ROS BAG
# -------------------------------
bridge = CvBridge()
images = []
tfs = []

with rosbag.Bag(bag_path) as bag:
    for topic, msg, t in bag.read_messages(topics=['/cf3/camera/image_raw', '/tf']):
        if topic == '/cf3/camera/image_raw':
            images.append((t.to_sec(), msg))
        elif topic == '/tf':
            tfs.append((t.to_sec(), msg))

def find_closest_tf(image_time, tf_list):
    return min(tf_list, key=lambda x: abs(x[0] - image_time))

def tf_to_homogeneous(transform):
    # Translation
    x, y, z = transform.translation.x, transform.translation.y, transform.translation.z
    # Rotation quaternion to rotation matrix
    r = Rotation.from_quat([transform.rotation.x,
                            transform.rotation.y,
                            transform.rotation.z,
                            transform.rotation.w])
    R = r.as_matrix()

    # Apply TSDF frame conversion (same as original code)
    xyz = np.array([-y, -z, x])
    M_change = np.array([[0,-1,0],[0,0,-1],[1,0,0]])
    R = M_change @ R @ M_change.T

    # Homogeneous matrix
    Hmtrx = np.hstack((R, xyz.reshape(3,1)))
    return np.vstack((Hmtrx, np.array([0,0,0,1])))

# -------------------------------
# PROCESS BAG AND SAVE DATA
# -------------------------------
for frame_number, (image_time, image_msg) in enumerate(images):
    # Convert ROS Image to OpenCV
    cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

    # Find closest TF
    closest_tf_time, closest_tf_msg = find_closest_tf(image_time, tfs)

    # Use the first transform in TFMessage
    transform = closest_tf_msg.transforms[0].transform

    # Convert to homogeneous matrix
    H = tf_to_homogeneous(transform)

    # Save image
    img_path = os.path.join(crazyflie_img_dir, f"crazyflie_frame-{frame_number:06d}.rgb.jpg")
    cv2.imwrite(img_path, cv_image)

    # Save pose
    pose_path = os.path.join(crazyflie_pose_dir, f"crazyflie_frame-{frame_number:06d}.pose.txt")
    np.savetxt(pose_path, H)

    # Display image
    cv2.imshow('crazyflie', cv_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("Done saving images and aligned poses from bag.")
