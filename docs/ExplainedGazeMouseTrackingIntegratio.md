# Mouse Tracking Application:

## What We've Built

We've created a sophisticated tracking application that records, analyzes, and visualizes how you use your mouse and where you look on screen.

## How It Works

The application has five main components:

1. **Core Mouse Tracker** (`mouse_tracker.py`): This is the module that captures every move, click, and scroll your mouse makes. It works silently in the background, recording data about where your mouse goes and what it does.

2. **Gaze Tracker** (`gaze_tracker.py`): This is the module that tracks where you're looking on your screen. It can use your webcam, a specialized Tobii eye tracker, or generate synthetic data for testing.

3. **Synchronization System** (`sync_tracker.py`): This is the module that connects mouse and gaze data, allowing you to see the relationship between where you look and where you click.

4. **Analytics Engine** (`mouse_analytics.py`): This is the module that processes all the recorded data and turns it into meaningful insights. It calculates statistics and creates visualizations to help you understand your mouse and gaze patterns.

5. **User Interface**: We provide two interface options:
   - **Mouse-Only Interface** (`mouse_tracker_gui.py`): A simpler interface focused just on mouse tracking
   - **Synchronized Interface** (`sync_tracker_gui.py`): The advanced interface that shows both mouse and gaze tracking together, with real-time visualization of their relationship

## Running the Application

You have multiple ways to run the application depending on your needs:

1. **Basic Mouse Tracking**:
   ```
   python mouse_tracker.py
   ```

2. **Mouse Tracking with GUI**:
   ```
   python mouse_tracker_gui.py
   ```

3. **Synchronized Mouse and Gaze Tracking** (Command Line):
   ```
   python sync_tracker.py --gaze-mode [webcam|tobii|dummy]
   ```

4. **Synchronized Tracking with Interactive GUI** (Recommended):
   ```
   python sync_tracker_gui.py
   ```

With the synchronized GUI, you can:
- Start and stop tracking sessions
- Clean the session to start fresh without restarting the application
- Switch between different gaze tracking modes
- Generate comprehensive reports
- See real-time visualizations of mouse movement, gaze tracking, and the relationship between them

## Clean Session Feature

The application now includes a "Clean Session" feature that allows you to:
- Reset your tracking data without stopping the application
- Start a new experiment immediately with the same settings
- Maintain your preferred tracking mode and settings between sessions
- Easily run multiple tracking sessions in sequence

This is particularly may be useful for researching purposes and conducting multiple experiments or when you want to discard practice runs and start with clean data.

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

## Visual Insights We'll Get

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

### Programming Concepts used in This Code

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

1. Integrate with specific applications to provide task-specific analysis
2. Extend to support other input devices like touchscreens and styluses
3. Develop interfaces for people with motor disabilities that respond to gaze 