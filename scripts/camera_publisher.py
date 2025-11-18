#!/usr/bin/env python3
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import threading

# Map each camera device to a drone
camera_devices = {
    "drone1": "/dev/video1",
    "drone2": "/dev/video2",
}

def publish_camera(drone_name, device):
    rospy.loginfo(f"[{drone_name}] Starting publisher on {device}")

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        rospy.logerr(f"[{drone_name}] Cannot open camera {device}")
        return
    else:
        rospy.loginfo(f"[{drone_name}] Camera {device} opened successfully")

    bridge = CvBridge()
    pub = rospy.Publisher(f"/{drone_name}/camera/image_raw", Image, queue_size=1)

    frame_count = 0
    rate = rospy.Rate(30)  # 30 Hz

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rospy.logwarn(f"[{drone_name}] Failed to read frame from {device}")
            rate.sleep()
            continue

        frame_count += 1
        msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        pub.publish(msg)

        if frame_count % 50 == 0:  # Log every 50 frames
            rospy.loginfo(f"[{drone_name}] Published {frame_count} frames")

        rate.sleep()

    cap.release()
    rospy.loginfo(f"[{drone_name}] Camera {device} released, publisher stopped")

if __name__ == "__main__":
    rospy.init_node("multi_drone_camera_publisher", log_level=rospy.INFO)
    threads = []

    for drone, dev in camera_devices.items():
        t = threading.Thread(target=publish_camera, args=(drone, dev), daemon=True)
        t.start()
        threads.append(t)

    rospy.loginfo("All camera publisher threads started")
    try:
        while not rospy.is_shutdown():
            rospy.sleep(1)
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down multi-drone camera publisher")
