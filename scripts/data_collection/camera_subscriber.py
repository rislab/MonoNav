#!/usr/bin/env python3
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# List of drone topics to subscribe to
drone_topics = [
    "/drone1/camera/image_raw",
    
]

bridge = CvBridge()

def image_callback(msg, window_name):
    try:
        # Convert ROS Image message to OpenCV image
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)
    except Exception as e:
        rospy.logerr(f"Failed to convert image: {e}")

if __name__ == "__main__":
    rospy.init_node("multi_drone_camera_viewer")

    for topic in drone_topics:
        # Use the topic name as window name
        rospy.Subscriber(topic, Image, image_callback, callback_args=topic)

    rospy.loginfo("Starting camera viewer. Press Ctrl+C to exit.")
    rospy.spin()
    cv2.destroyAllWindows()
