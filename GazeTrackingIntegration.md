# Integrated Mouse and Gaze Tracking System

This document explains the advanced gaze tracking component that has been integrated with the mouse tracking system. The integration allows for synchronous tracking of both mouse movements and eye gaze, providing valuable insights into the relationship between visual attention and mouse control.

## Gaze Tracking Technologies

The system supports three different methods for gaze tracking:

1. **Webcam-based tracking** - Uses computer vision techniques to detect eye movements from a standard webcam
2. **Tobii eye tracker** - Supports professional-grade eye tracking hardware for maximum precision
3. **Dummy tracking** - Simulates gaze data for testing and development purposes

## How Gaze Tracking Works

### Webcam-based Tracking

The webcam-based approach uses the following steps:

1. **Face detection** - The system first locates the user's face in the webcam feed
2. **Facial landmark detection** - It then identifies 68 key points on the face, including eye corners and eyelids
3. **Eye region extraction** - The regions around the eyes are isolated for detailed analysis
4. **Pupil detection** - Computer vision techniques (thresholding and contour detection) identify the dark pupil
5. **Gaze estimation** - The position of the pupils relative to the eye corners is used to estimate where the user is looking
6. **Calibration** - A mapping is created between pupil positions and screen coordinates

### Gaze Events

The system tracks three types of gaze events:

1. **Fixations** - When the eye remains relatively stable, focusing on a specific point
2. **Saccades** - Rapid movements between fixations
3. **Blinks** - Eye closure events

## Data Synchronization

The `SyncTracker` module is the heart of the integrated system, performing several key functions:

1. **Event correlation** - It matches up temporally close mouse and gaze events
2. **Position alignment** - It associates mouse positions with gaze positions occurring at approximately the same time
3. **Distance calculation** - It computes the distance between mouse cursor and estimated gaze point
4. **Attention matching** - It determines when the user's gaze is following the mouse cursor

## Analytics and Visualizations

The synchronized tracking enables several advanced analytics:

1. **Attention match percentage** - Percentage of time the user's gaze is near the mouse cursor
2. **Distance metrics** - Average and maximum distance between gaze and cursor
3. **Heatmap comparison** - Visual comparison of where the mouse moves versus where the user looks
4. **Trajectory visualization** - Path visualization showing how eye movements relate to mouse movements
5. **Time-synchronized data** - All events are timestamped allowing for temporal analysis

## Scientific Applications

This integrated tracking system can be used for various scientific and UX research purposes:

1. **Usability testing** - Understand how users visually process interfaces before interacting
2. **Cognitive load assessment** - Large gaze-mouse distances may indicate higher cognitive load
3. **Attention patterns** - Identify what users look at vs. what they interact with
4. **Interaction efficiency** - Measure how directly users navigate to targets

## Technical Requirements

For webcam-based tracking:
- Webcam (higher resolution provides better accuracy)
- Good, consistent lighting
- Face landmark model (provided by dlib)

For Tobii tracking:
- Tobii eye tracker hardware
- Tobii Research SDK

## Environment Variables

The system can be configured using environment variables:

- `MOUSE_TRACKER_DATA_DIR`: Directory to save tracking data
- `MOUSE_TRACKER_DEBUG`: Enable debug logging (true/false)
- `GAZE_TRACKER_MODE`: Set the tracking mode ('webcam', 'tobii', or 'dummy')

## Getting Started

To use the integrated tracking system:

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. For webcam-based tracking, download the face landmark model:
   ```bash
   # Create models directory
   mkdir -p models
   
   # Download and extract the model (requires wget and bzip2)
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
   mv shape_predictor_68_face_landmarks.dat models/
   ```

3. Run the synchronous tracker:
   ```bash
   python sync_tracker.py --gaze-mode webcam --report
   ```

4. When finished, press Ctrl+C to stop tracking and generate a report. 