# Mouse Tracking Application: A Simple Explanation

## What We've Built

We've created a sophisticated tracking application that records, analyzes, and visualizes how you use your mouse and where you look on screen. Think of it like a fitness tracker, but for your mouse movements and eye gaze instead of your steps!

## How It Works (The Simple Version)

The application has five main components:

1. **Core Mouse Tracker** (`mouse_tracker.py`): This is the "brain" that captures every move, click, and scroll your mouse makes. It works silently in the background, recording data about where your mouse goes and what it does.

2. **Gaze Tracker** (`gaze_tracker.py`): This is the "eye" that tracks where you're looking on your screen. It can use your webcam, a specialized Tobii eye tracker, or generate simulated data for testing.

3. **Synchronization System** (`sync_tracker.py`): This is the "coordinator" that connects mouse and gaze data, allowing you to see the relationship between where you look and where you click.

4. **Analytics Engine** (`mouse_analytics.py`): This is the "analyst" that processes all the recorded data and turns it into meaningful insights. It calculates statistics and creates visualizations to help you understand your mouse and gaze patterns.

5. **User Interface** (`mouse_tracker_gui.py`): This is the "face" of the application - what you actually see and interact with. It provides buttons to start/stop tracking and shows real-time visualizations.

## What It Actually Does

When you run the synchronized tracking application:

1. It tracks **every movement** of your mouse across your screen
2. It records **every click** you make (left, right, and middle button)
3. It monitors **every scroll** action
4. It tracks **where your eyes are looking** on the screen
5. It identifies **fixations** (when your gaze stays in one place), **saccades** (rapid eye movements), and **blinks**
6. It calculates the **relationship** between your mouse position and gaze position
7. It saves all this data to files for later analysis
8. It calculates metrics like:
   - Total distance your mouse travels
   - Average distance between mouse and gaze points
   - Attention match percentage (how often you're looking at or near your cursor)
   - Areas of your screen where your eyes and mouse spend the most time

## How Gaze Tracking Works

The application offers three methods of tracking where you look:

1. **Webcam**: Using your regular computer webcam, the application:
   - Finds your face in the video feed
   - Locates your eyes using facial landmarks
   - Tracks your pupils to estimate where you're looking
   - Maps this to screen coordinates

2. **Tobii Eye Tracker**: Using specialized hardware for professional tracking:
   - Connects to Tobii eye tracking devices
   - Receives precise gaze coordinates directly from the device
   - Provides high-accuracy data for research purposes

3. **Dummy Mode**: For testing or when no hardware is available:
   - Generates realistic simulated gaze data
   - Mimics typical eye movement patterns (fixations, saccades, blinks)
   - Allows you to test the software without tracking hardware

## Visual Insights You'll Get

The synchronized tracking generates powerful visualizations:

1. **Heatmap Comparison**: See where your mouse moves versus where your eyes look
2. **Distance Plot**: Graph showing how far apart your gaze and cursor are over time
3. **Trajectory Comparison**: Visual paths showing mouse movements and eye movements
4. **HTML Reports**: Comprehensive analysis with all these visuals plus metrics

## How It Works With Your Data

The application is designed with data privacy and flexibility in mind:

1. **Local Storage**: All data stays on your computer - nothing is sent to external servers
2. **Customizable Storage**: You can choose where to save data using the environment variable `MOUSE_TRACKER_DATA_DIR`
3. **Multiple Formats**: Data is saved in both CSV (spreadsheet-friendly) and JSON (programming-friendly) formats
4. **Data Visualization**: The application creates visual reports to help you understand the data
5. **Data Control**: You can delete the data anytime by removing files from the data directory

## Technical Insights

### Programming Concepts You Can Learn From This Code

1. **Object-Oriented Programming**: The application uses classes to organize code logically

2. **Event-Driven Programming**: The application responds to mouse and gaze events rather than executing in a linear sequence

3. **Multithreading**: The trackers run on separate threads so they can record events while the rest of the application stays responsive

4. **Computer Vision**: The webcam gaze tracker uses techniques like facial landmark detection and pupil tracking

5. **Data Processing with Pandas**: The analytics module processes and analyzes large datasets efficiently

6. **Data Visualization**: The application creates informative visualizations from raw data

7. **Hardware Integration**: The application can interface with specialized eye tracking hardware

8. **Environment Variables**: The application uses environment variables for configuration

### Design Patterns

1. **Observer Pattern**: The trackers observe events and notify the application when they occur

2. **Factory Pattern**: Different tracker types (webcam, Tobii, dummy) are created based on configuration

3. **Adapter Pattern**: Different data sources (mouse, webcam, eye tracker) are adapted to a common interface

4. **Singleton Pattern**: The core trackers ensure only one instance is tracking at any time

## Why This Matters

Combined mouse and gaze tracking can be useful for:

1. **UX Research**: See what users look at before they click, revealing attention patterns
2. **Usability Testing**: Identify confusing interfaces where users look in many places before finding what they need
3. **Cognitive Load Assessment**: Large distances between gaze and mouse may indicate higher cognitive load
4. **Accessibility Research**: Understand how people with different abilities interact with interfaces
5. **Human-Computer Interaction Studies**: Research how visual attention relates to physical interaction

## Next Steps

To take this project further, you could:

1. Add machine learning to identify patterns in the relationship between gaze and mouse movement
2. Create a real-time feedback system for improving coordination between eyes and mouse
3. Integrate with specific applications to provide task-specific analysis
4. Extend to support other input devices like touchscreens and styluses
5. Develop interfaces for people with motor disabilities that respond to gaze 