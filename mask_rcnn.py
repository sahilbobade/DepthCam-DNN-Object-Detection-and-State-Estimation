

import cv2
import numpy as np
import math
class MaskRCNN:
    def __init__(self):
        # Loading Mask RCNN
        self.net = cv2.dnn.readNetFromTensorflow("/home/sahil/catkin_ws/src/test/scripts/dnn/frozen_inference_graph_coco.pb",
                                                 "/home/sahil/catkin_ws/src/test/scripts/dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        np.random.seed(2)
        self.colors = np.random.randint(0, 255, (90, 3))
        self.detection_threshold = 0.7
        self.mask_threshold = 0.3
        self.classes = []
        with open("/home/sahil/catkin_ws/src/test/scripts/dnn/classes.txt", "r") as file_object:
            for class_name in file_object.readlines():
                self.classes.append(class_name.strip())
        self.obj_boxes = []
        self.obj_classes = []
        self.obj_centers = []
        self.obj_contours = []
        self.target_class_ids = {0, 43}
    def detect_objects_mask(self, bgr_frame):
        blob = cv2.dnn.blobFromImage(bgr_frame, swapRB=True)
        self.net.setInput(blob)
        boxes, masks = self.net.forward(["detection_out_final", "detection_masks"])
        frame_height, frame_width, _ = bgr_frame.shape
        detection_count = boxes.shape[2]
        self.obj_boxes.clear()
        self.obj_classes.clear()
        self.obj_centers.clear()
        self.obj_contours.clear()
        for i in range(detection_count):
            box = boxes[0, 0, i]
            class_id = box[1]
            score = box[2]
            if score < self.detection_threshold or class_id not in self.target_class_ids:
                continue
            x = int(box[3] * frame_width)
            y = int(box[4] * frame_height)
            x2 = int(box[5] * frame_width)
            y2 = int(box[6] * frame_height)
            self.obj_boxes.append([x, y, x2, y2])
            cx = (x + x2) // 2
            cy = (y + y2) // 2
            self.obj_centers.append((cx, cy))
            self.obj_classes.append(class_id)
            mask = masks[i, int(class_id)]
            roi_height, roi_width = y2 - y, x2 - x
            mask = cv2.resize(mask, (roi_width, roi_height))
            _, mask = cv2.threshold(mask, self.mask_threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.obj_contours.append(contours)
        # Make sure to return the accumulated results
        return self.obj_boxes, self.obj_classes, self.obj_contours, self.obj_centers
    def draw_object_mask(self, bgr_frame):
        for box, class_id, contours in zip(self.obj_boxes, self.obj_classes, self.obj_contours):
            x, y, x2, y2 = box
            roi = bgr_frame[y: y2, x: x2]
            color = tuple([int(c) for c in self.colors[int(class_id)]])  # Ensure color is an integer tuple
            roi_copy = np.zeros_like(roi)
            for cnt in contours:
                cv2.drawContours(roi, [cnt], -1, color, 3)
                cv2.fillPoly(roi_copy, [cnt], color)
                roi = cv2.addWeighted(roi, 1, roi_copy, 0.5, 0.0)
                bgr_frame[y: y2, x: x2] = roi
            # Calculate the center and the radius for the circle
            center = (x + (x2 - x) // 2, y + (y2 - y) // 2)
            radius = max(x2 - x, y2 - y) // 2
            cv2.circle(bgr_frame, center, radius, color, 2)  # Draw the circle
        return bgr_frame
    def draw_object_info(self, bgr_frame, depth_frame):
        for box, class_id, obj_center in zip(self.obj_boxes, self.obj_classes, self.obj_centers):
            x, y, x2, y2 = box
            color = tuple([int(c) for c in self.colors[int(class_id)]])  # Ensure color is an integer tuple
            cx, cy = obj_center
            depth_mm = depth_frame[cy, cx]
            cv2.rectangle(bgr_frame, (x, y), (x2, y2), color, 1)
            print(class_id)
            class_name = self.classes[int(class_id)]
            cv2.putText(bgr_frame, class_name.capitalize(), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(bgr_frame, "{} cm".format(depth_mm / 10), (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                        2)

    def draw_object_size(self, bgr_frame, depth_frame, fov_x, fov_y):
        for box, class_id, obj_center in zip(self.obj_boxes, self.obj_classes, self.obj_centers):
            x, y, x2, y2 = box
            width_px = x2 - x
            height_px = y2 - y
            depth_mm = depth_frame[obj_center[1], obj_center[0]]
            depth_m = depth_mm / 1000.0  # Convert depth to meters for calculation
            # Calculate the size of the object in meters
            width_m = (2 * (depth_m * math.tan(fov_x / 2)) * (width_px / 640))
            height_m = (2 * (depth_m * math.tan(fov_y / 2)) * (height_px / 480))
            color = self.colors[int(class_id)]
            # Ensure color is an integer tuple for cv2.putText
            color = tuple([int(c) for c in color])
            cv2.putText(bgr_frame, "W: {:.2f}m, H: {:.2f}m".format(width_m, height_m),
                        (x, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return bgr_frame