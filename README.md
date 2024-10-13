# kalman-object-tracker
# Kalman Tracker Command-Line App

## Overview

The **Kalman Tracker Command-Line App** is  tracking system designed for multi-object tracking using the Kalman filter. It tracks objects in a video stream or image sequence by using detected bounding boxes from YOLO11s and Kalman filters to estimate and predict their positions over time. This application can be used in various scenarios such as surveillance, vehicle tracking, and retail monitoring.

## Features

- **Single and Multi-object tracking**: Supports tracking of multiple objects in real-time.
- **Kalman filter-based tracking**: Uses Kalman filters for smooth prediction and correction of object positions.
- **Euclidean & Mahalanobis distance matching**: Dynamically matches detections to tracked objects using different distance metrics.
- **Prediction for missed detections**: Continues tracking even if an object is momentarily lost in detection.
- **Acceleration model support**: Optionally supports an acceleration model for smoother tracking of objects in motion.

## How It Works

1. **Object Detection**: The app relies on an external object detection system (YOLO11 but can be customized for other models or methods) to provide detected bounding boxes.
2. **Tracking**: Once detections are provided, the Kalman filter predicts the object's next position based on its past states. The app updates the tracked object’s state when new detections are available or predicts their position if no detection is found.
3. **Matching**: Detected objects are matched to previously tracked objects using Euclidean or Mahalanobis distances.
4. **Frame Update**: The tracked object’s positions are drawn on each frame of the video stream or image sequence with a circle in the center. The circle have 3 colors indicating the state of the tracked object:
- **Green**: The object is detected and tracked
- **Orange**: The object is lost (not detected but tracked)
- **Red**: The object will be considered lost and won't be tracked any further if it doesn't appear again in the next frame.

## Dependencies

This application depends on the following Python packages:

- `opencv-python`: Used for handling video streams, drawing bounding boxes, and displaying results.
- `numpy`: Used for numerical computations, including Kalman filter operations.
- `scipy`: Used for calculating distances between tracked objects and detections.
- `matplotlib`: Used for plotting the estimated variables
- `ultralytics`: Used for detecting the objects using yolo11. 

To install the required dependencies, run:

```bash
# create a virtual environment
python -m venv .tracker
# install dependencies
pip install -r requirements.txt
````

## Running the app
```bash
# show all possible options
python track.py --help

python track.py --mode single --video-source 0
````

## System Design
![ System Design]("./docs/system_design.png")


## Justification of choices made

### Model : YOLO11s
- Light-weight and fast
- Can be used for real-time applications like object tracking
### Ultralytics
- Provides pre-trained yolo models with the possibility to deploy them to multiple platform.
- YOLO exported on ONNx format gives x3 speed in cpu.
### Tracker: Kalman
- Easy to understand and implement
- Robust in tracking objects moving at constant speed
- Tracker can later be extended to deepsort for more complex tracking as it rely on Kalman Filter.
### Object Association method: Distance
- Euclidean Fast and simple and works well in environments with less uncertainties. Works better than Mahalanobis when the covariance between points is not known.
- Mahalanobis woks best in environment with non-uniform uncertainties. But can't be reliable if the covariance between points is set randomly.
When using Kalman Filter the residual covariance could be passed to this distance.

### Object state management:
- To better monitor each object and track where it has lastly been seen.
### Use of Cache: JSON
- Cache: Necessary for analysis and in case of interruptions
- JSON is used only for mocking purposes.

### Application: Command Line
- Light weight
- serves well for an MVP

## Estimated velocities and displacements on the example videos
![ Persons]("results\movement_video_senators.png")

