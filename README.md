# DepthCam-DNN-Object-Detection-and-State-Estimation

---

# Object Detection and Size Estimation with Intel RealSense

## Overview
This project utilizes the Intel RealSense camera to detect objects within its field of view, estimate their sizes, and provide precise coordinates. It is designed for applications requiring spatial awareness and object interaction, making it ideal for robotics, automated navigation systems, and environmental monitoring.

## Features
- **Real-Time Object Detection**: Leverages the advanced capabilities of the Intel RealSense camera to detect objects in real-time.
- **Size Estimation**: Estimates the dimensions of detected objects, providing essential data for various applications.
- **Coordinate Mapping**: Calculates and outputs the coordinates of each detected object relative to the camera’s position.

## Technology
This project is developed using:
- Intel RealSense Camera
- ROS
- Libresense
- OpenCV

## Requirements

### Hardware Requirements
- Intel RealSense Camera

### Software Requirements
- Python (recommended version 3.8.10)
- `pyrealsense2` for running Python scripts with Intel RealSense support
- RealSense ROS wrapper for running ROS nodes
- OpenCV (Python library): `opencv-python`
- NumPy


## Getting Started
Follow these instructions to set up and run the project:
1. **Prerequisites**: Ensure you have the necessary hardware and software installed.
2. **Installation**: Clone this repository and install the required dependencies listed in the `requirements.txt` file.
3. **Running the Application**: Detailed steps on how to run the application are in below section.

Here's how you can structure the instructions in your `README.md` to guide users on how to use the scripts for both methods:

---

## Usage Instructions

This project provides two methods for running the object detection and size estimation tasks using the Intel RealSense camera.

### Method 1: Running Python Code (ROS not required)

Follow these steps to run the object detection without ROS:

1. **Connect the Intel RealSense Camera**: Ensure that the camera is connected to your computer.
2. **Run the Python Script**: Execute the script by running the following command in your terminal:
   ```
   python Python_script.py
   ```

### Method 2: Running ROS Node

If you are using ROS to run the application, follow these steps:

1. **Launch Intel RealSense Node**: Start the RealSense camera node using the following command in your terminal:
   ```
   roslaunch realsense2_camera rs_camera.launch
   ```
2. **Connect the Intel RealSense Camera**: Ensure that the camera is connected to your computer.
3. **Run the ROS Node**: Execute the ROS node by running:
   ```
   rosrun your_package_name ROS_Node.py
   ```

   Replace `your_package_name` with the name of your ROS package where `ROS_Node.py` is located.

Ensure you have the necessary permissions to execute the scripts and that all dependencies are installed as per the `requirements.txt` and ROS setup documentation.


## Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**. Please refer to the CONTRIBUTING.md for guidelines on how to proceed.

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
Sahil Bobade – sahilbobade751@gmail.com
Project Link: https://github.com/sahilbobade/DepthCam-DNN-Object-Detection-and-State-Estimation

---
