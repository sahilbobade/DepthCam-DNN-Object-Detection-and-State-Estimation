import cv2
import math
from realsense_camera import *
from mask_rcnn import *
import pyrealsense2
# Load Realsense camera
rs = RealsenseCamera()
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

while True:
    # Get frame in real time from Realsense camera
    ret, bgr_frame, depth_frame = rs.get_frame_stream()

    if not ret:
        print("No frame received from the camera. Exiting...")
        break

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
            print(result[0]/100,result[2]/10)

    cv2.imshow("Depth Frame", depth_frame)
    cv2.imshow("BGR Frame", bgr_frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key to break
        break

rs.release()
cv2.destroyAllWindows()
