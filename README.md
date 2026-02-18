# Soccer Analysis Project
## Introduction

The Soccer Analysis Project is a computer vision–based system designed to detect, track, and analyze players, referees, and the football in match videos using state-of-the-art AI techniques.

The project leverages YOLO object detection, clustering algorithms, optical flow, and perspective transformation to extract meaningful match insights such as:

- Player tracking

- Team classification

- Ball possession percentage

- Speed and distance covered per player

- Movement analysis in real-world (meter) scale

## Core Technologies & Modules
### YOLO – Object Detection

- Used for detecting Players, Referees and Ball

### KMeans – Team Classification

- Extracts dominant jersey colors
- Clusters players into teams based on t-shirt color
- Automatically assigns team IDs

### Optical Flow – Camera Motion Estimation

- Detects camera movement between frames
- Compensates tracking to improve accuracy
- Ensures real player movement is measured correctly

### Perspective Transformation

- Converts pixel coordinates into real-world scale
- Enables measuring movement in meters instead of pixels

### Speed & Distance Estimation

- Calculates Distance covered per player, Average speed and Instantaneous speed


## Requirements

- Python
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas
