#!/usr/bin/env python3

import rospy
import cv2
import math
from cv_bridge import CvBridge, CvBridgeError
import pyrealsense2
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import sys
sys.path.append('/home/sahil/catkin_ws/src/test/scripts/')
from mask_rcnn import MaskRCNN  
from realsense_camera import *

mrcnn = MaskRCNN()

_intrinsics = pyrealsense2.intrinsics()
_intrinsics.width = 640
_intrinsics.height = 480
_intrinsics.ppx = 320
_intrinsics.ppy = 240
_intrinsics.fx = 69.4
_intrinsics.fy = 42.5
_intrinsics.model  = pyrealsense2.distortion.none
_intrinsics.coeffs = [0, 0, 0, 0, 0]

mrcnn = MaskRCNN()
class RealSenseROSNode:
    def __init__(self):
        rospy.init_node('realsense_ros_node', anonymous=True)
        
        self.bridge = CvBridge()
        self.mrcnn = MaskRCNN()
        
        # Subscribers
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        self.coord_pub = rospy.Publisher('coordinates', Float32MultiArray, queue_size=10)
        
        # Internal state
        self.current_color_image = None
        self.current_depth_image = None
        self.coord_array = Float32MultiArray()

        # Camera intrinsics (assuming these values are constant)
        self.intrinsics = {
            'width': 640,
            'height': 480,
            'ppx': 320,
            'ppy': 240,
            'fx': 69.4,
            'fy': 42.5,
            'model': 'none',
            'coeffs': [0, 0, 0, 0, 0]
        }
        self.fov_x = math.radians(69.4)
        self.fov_y = math.radians(42.5)
        print('init completed')

    def image_callback(self, data):
        try:
            self.current_color_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            #print('imag callback')
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def depth_callback(self, data):
        try:
            self.current_depth_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
            #print('depth callback')
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def deproject_pixel_to_point(self, pixel, depth):
        x, y = pixel
        depth = float(depth) / 1000.0  # Convert depth to meters
        x = (x - self.intrinsics['ppx']) / self.intrinsics['fx']
        y = (y - self.intrinsics['ppy']) / self.intrinsics['fy']
        return [depth * x, depth * y, depth]


    def process_images(self):
        if self.current_color_image is not None and self.current_depth_image is not None:
            #print('inside process images')
            bgr_frame = self.current_color_image
            depth_frame = self.current_depth_image

            # Get object mask
            boxes, classes, contours, centers = mrcnn.detect_objects_mask(bgr_frame)

            # Draw object mask
            bgr_frame = mrcnn.draw_object_mask(bgr_frame)

            # Show depth info and size of the objects
            mrcnn.draw_object_info(bgr_frame, depth_frame)
            fov_x, fov_y = math.radians(69.4), math.radians(42.5)  # Use the correct FOV values for your camera model
            bgr_frame = mrcnn.draw_object_size(bgr_frame, depth_frame, fov_x, fov_y)

            # Iterate over detected objects
            for box, cls, ctr in zip(boxes, classes, centers):
                #print(cls)
                if cls == 43:
                    # Calculate centroid
                    centroid_x = (box[0] + box[2]) // 2
                    centroid_y = (box[1] + box[3]) // 2
                    centroid_depth = depth_frame[centroid_y, centroid_x]

                    # Print centroid coordinates and depth
                    print("Bottle centroid (x, y, depth):", centroid_x, centroid_y, centroid_depth)

                    # Draw centroid on the frame
                    cv2.circle(bgr_frame, (centroid_x, centroid_y), 5, (0, 255, 0), -1)

                    result = pyrealsense2.rs2_deproject_pixel_to_point(_intrinsics, [centroid_x, centroid_y], centroid_depth)
                    self.coord_array.data = [result[0]/100,centroid_depth/10] 
                    rospy.loginfo(f"Publishing coordinates: x={result[0]/100}, y={centroid_depth/10}")
                    self.coord_pub.publish(self.coord_array)
                    # print(result[0]/100,result[2]/10)

            cv2.imshow("Depth Frame", depth_frame)
            cv2.imshow("BGR Frame", bgr_frame)

            key = cv2.waitKey(1)

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            self.process_images()
            rate.sleep()

        cv2.destroyAllWindows()

if __name__ == '__main__':
    node = RealSenseROSNode()
    node.run()
