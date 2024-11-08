# YOLO Python Scripts Collection

This repository contains multiple Python scripts utilizing the YOLO model for various real-time object detection and pose estimation tasks, including fall detection and general object recognition.

## Overview
The scripts here use the YOLO model from the [Ultralytics library](https://github.com/ultralytics/ultralytics) for detecting objects and poses in video feeds, primarily via webcam. YOLO (You Only Look Once) is an advanced, high-speed object detection model, making it suitable for real-time applications.

## Features
- **Pose Detection**: Detects human poses and highlights keypoints to monitor body orientation, with fall detection alerts when unusual body posture is identified.
- **Object Recognition**: Identifies various object classes (e.g., person, vehicle, animals) in real-time video.
- **Customizable Thresholds**: Adjusts confidence thresholds for object detection and classification.

## Requirements
- Python
- OpenCV
- Ultralytics YOLO library

## Usage
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install ultralytics opencv-python
