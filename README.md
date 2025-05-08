# Advanced Mouse Tracking Application

This is an advanced Python application for tracking and analyzing mouse movement, clicks, and scroll events with comprehensive visual analytics.

## Features

- **Real-time mouse tracking** - Captures x, y coordinates, clicks, and scroll events
- **Movement visualization** - Shows mouse trajectories and heatmaps
- **Click analysis** - Records and analyzes mouse button clicks
- **Comprehensive metrics** - Distance moved, speed, acceleration, and more
- **Data export** - Saves to CSV and JSON formats
- **Visual reports** - Generates HTML reports with interactive visualizations
- **User-friendly GUI** - Easy to use interface with real-time visualization

## Advanced Gaze Tracking

This application now includes advanced gaze tracking capabilities that can be used in conjunction with mouse tracking. The gaze tracking supports three different modes:

1. **Webcam-based tracking** - Uses your webcam to estimate where you're looking on screen
2. **Tobii eye tracker** - Compatible with Tobii eye tracking hardware for precise measurements
3. **Dummy tracking** - Simulates gaze data for testing and development

### Synchronized Mouse and Gaze Tracking

The `sync_tracker.py` module provides synchronized tracking of both mouse and gaze movements, enabling:

- Comparison of where you look versus where you click
- Analysis of the coordination between visual attention and mouse control
- Visualization of attention patterns during computer use

### Running Gaze Tracking

To run the synchronized mouse and gaze tracker:

```bash
# Using webcam-based tracking
python sync_tracker.py --gaze-mode webcam --report

# Using dummy mode (no hardware required)
python sync_tracker.py --gaze-mode dummy --report

# Using Tobii hardware (if available)
python sync_tracker.py --gaze-mode tobii --report
```

For webcam-based tracking, you'll need to download the face landmark model:

```bash
# Create models directory
mkdir -p models

# Download and extract the model (requires wget and bzip2)
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
mv shape_predictor_68_face_landmarks.dat models/
```

### Environment Variables

Configure the gaze tracking using environment variables:

- `GAZE_TRACKER_MODE`: Set tracking mode ('webcam', 'tobii', or 'dummy')

### Additional Requirements

For gaze tracking, additional dependencies are required:

- OpenCV for computer vision processing
- dlib for face and landmark detection
- scipy for signal processing

These are included in `requirements.txt`.

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd MouseTracking
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Run the basic mouse tracker from the command line:

```bash
python mouse_tracker.py --output mouse_data --duration 60
```

Parameters:
- `--output`: Directory to save tracking data (default: "mouse_data")
- `--duration`: Duration to track in seconds (0 for indefinite, press Ctrl+C to stop)

### Graphical User Interface

There are two GUI options available:

1. **Mouse-only tracking GUI**:
```bash
python mouse_tracker_gui.py
```

2. **Synchronized mouse and gaze tracking GUI** (recommended):
```bash
python sync_tracker_gui.py
```

The synchronized tracking GUI provides:
- Start/stop tracking controls for both mouse and gaze
- Clean Session feature to reset data without restarting
- Real-time visualization of mouse movement and gaze points
- Live statistics on both mouse and gaze events
- Distance analysis between gaze and mouse positions
- Easy report generation

### Clean Session Feature

The synchronized tracking GUI includes a "Clean Session" button that allows you to:
- Reset all tracking data without closing the application
- Start a new tracking session with the same settings
- Maintain your tracking mode between sessions
- Run consecutive experiments easily

This is particularly useful for researchers conducting multiple tests or when you want to discard practice data.

### Data Analysis

To analyze previously recorded data:

```bash
python mouse_analytics.py --data-dir mouse_data --report
```

Parameters:
- `--data-dir`: Directory containing mouse tracking data
- `--report`: Generate a comprehensive HTML report

## Output Files

The application generates several file types:

- **CSV files**: Raw data in comma-separated format
- **JSON files**: Structured data for easy parsing
- **PNG files**: Visualizations (heatmaps, trajectories, etc.)
- **HTML reports**: Comprehensive analytics reports

## Project Structure

- `mouse_tracker.py`: Core mouse tracking functionality
- `gaze_tracker.py`: Core gaze tracking functionality
- `sync_tracker.py`: Synchronized mouse and gaze tracking
- `mouse_analytics.py`: Data analysis and visualization
- `mouse_tracker_gui.py`: Basic graphical user interface
- `sync_tracker_gui.py`: Advanced synchronized tracking interface
- `requirements.txt`: Python dependencies

## Example Analysis

After tracking your mouse movements, you can generate reports that include:

- Mouse movement heatmaps showing where your cursor spends the most time
- Trajectory plots showing the path of your mouse
- Click distribution analysis
- Speed and acceleration metrics
- Comprehensive statistics

## Requirements

- Python 3.7 or higher
- Dependencies listed in requirements.txt

## License

This project is released under the MIT License.

## GitHub Repository Setup

To set up a GitHub repository for this project:

1. Create a new repository on GitHub:
   - Go to https://github.com/new
   - Name your repository (e.g., "MouseTracking")
   - Choose public or private visibility
   - Do not initialize with README, .gitignore, or license (we already have these)
   - Click "Create repository"

2. Initialize your local repository and push to GitHub:
```bash
# Initialize the repository
git init

# Add all files
git add .

# Commit the files
git commit -m "Initial commit"

# Connect to your GitHub repository
git remote add origin https://github.com/YOUR_USERNAME/MouseTracking.git

# Push to GitHub
git push -u origin main
```

## Environment Setup

The project includes scripts to set up your environment:

### For macOS/Linux:
```bash
# Make the script executable
chmod +x setup_env.sh

# Run the setup script
./setup_env.sh
```

### For Windows:
```bash
# Run the setup script
setup_env.bat
```

These scripts will:
1. Create a virtual environment
2. Install all dependencies
3. Set up required environment variables:
   - `MOUSE_TRACKER_DATA_DIR`: Directory to store tracking data
   - `MOUSE_TRACKER_DEBUG`: Enable debug mode (true/false)

## Environment Variables

You can customize the application behavior with these environment variables:

- `MOUSE_TRACKER_DATA_DIR`: Path where tracking data will be saved (default: ./mouse_data)
- `MOUSE_TRACKER_DEBUG`: Enable debug logging when set to 'true' (default: false)

To set these manually:

### For macOS/Linux:
```bash
export MOUSE_TRACKER_DATA_DIR="/path/to/data"
export MOUSE_TRACKER_DEBUG=true
```

### For Windows:
```bash
set MOUSE_TRACKER_DATA_DIR=C:\path\to\data
set MOUSE_TRACKER_DEBUG=true
``` 