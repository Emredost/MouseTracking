# MouseTracking Application: Architecture Documentation

## Table of Contents
- [Application Overview](#application-overview)
- [Folder Structure](#folder-structure)
- [Core Features](#core-features)
- [Design Decisions](#design-decisions)
- [Best Practices](#best-practices)
- [OS Compatibility](#os-compatibility)
- [Libraries and Dependencies](#libraries-and-dependencies)
- [Data Flow](#data-flow)
- [Future Development](#future-development)

## Application Overview

The MouseTracking application is a comprehensive system for tracking, analyzing, and visualizing mouse movements and gaze patterns simultaneously. It's designed for researchers, UX professionals, and developers who need to understand how users interact with interfaces through both mouse movement and eye gaze.

The application follows a modular architecture with clear separation of concerns between tracking, analysis, and visualization components.

## Folder Structure

```
MouseTracking/
├── mouse_tracker.py          # Core mouse tracking functionality
├── gaze_tracker.py           # Core gaze tracking with webcam/Tobii/dummy modes
├── sync_tracker.py           # Synchronization of mouse and gaze data
├── mouse_analytics.py        # Data analysis and visualization tools
├── mouse_tracker_gui.py      # Basic mouse-only tracking interface
├── sync_tracker_gui.py       # Advanced synchronized tracking interface
├── run_sync_tracker.py       # Easy entry point with auto-setup
├── models/                   # Directory for facial landmark models
│   └── shape_predictor_68_face_landmarks.dat  # dlib facial landmark model
├── mouse_data/               # Output directory for tracking data
│   ├── mouse_events_*.csv    # Mouse event data in CSV format
│   ├── mouse_events_*.json   # Mouse event data in JSON format
│   ├── gaze_events_*.csv     # Gaze event data in CSV format
│   ├── gaze_events_*.json    # Gaze event data in JSON format
│   ├── sync_events_*.csv     # Synchronized event data in CSV format
│   ├── sync_events_*.json    # Synchronized event data in JSON format
│   └── *_report_*.html       # Generated HTML reports with visualizations
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview and usage instructions
├── MouseTracker_Explained.md # Simplified explanation of the system
├── .gitignore                # Git exclusion patterns
├── setup_env.sh              # Environment setup for Unix/macOS
└── setup_env.bat             # Environment setup for Windows
```

## Core Features

### 1. Mouse Tracking
- **Feature**: Real-time tracking of mouse movements, clicks, and scrolls
- **Why**: Mouse movements provide essential data about user interaction patterns and help identify usability issues
- **Implementation**: Uses the `pynput` library for cross-platform mouse event capturing
- **Benefits**: Non-intrusive tracking that works with any application

### 2. Gaze Tracking (Three Modes)
- **Feature**: Tracks where the user is looking on screen
- **Why**: Gaze data reveals attention patterns and visual focus that mouse data alone cannot capture
- **Implementation**:
  - **Webcam Mode**: Uses computer vision to estimate gaze direction based on eye position
  - **Tobii Mode**: Integrates with specialized Tobii eye tracking hardware for precision
  - **Dummy Mode**: Generates synthetic gaze data for testing without hardware
- **Benefits**: Flexible options for different research needs and hardware availability

### 3. Synchronized Tracking
- **Feature**: Combines mouse and gaze data in a time-synchronized way
- **Why**: Allows correlation between where users look and where they click
- **Implementation**: Uses timestamp alignment and calculates metrics like gaze-mouse distance
- **Benefits**: Reveals patterns in the coordination between visual attention and physical interaction

### 4. Real-time Visualization
- **Feature**: Live plotting of movements, trajectories, and statistical data
- **Why**: Immediate feedback helps identify patterns during the session
- **Implementation**: Matplotlib integration with Tkinter for responsive UI
- **Benefits**: No need to wait for post-processing to see results

### 5. Clean Session Feature
- **Feature**: Reset tracking without restarting the application
- **Why**: Enables consecutive experiments without interruption
- **Implementation**: Reinitializes trackers while maintaining settings
- **Benefits**: Improves workflow for researchers conducting multiple tests

### 6. Comprehensive Reporting
- **Feature**: Generates detailed HTML reports with interactive visualizations
- **Why**: Facilitates deeper analysis and sharing of results
- **Implementation**: Custom report generation with embedded visualizations
- **Benefits**: Easy to share findings with stakeholders

## Design Decisions

### Modular Architecture
- **Decision**: Separate tracking, analysis, and visualization into distinct components
- **Why**: Enhances maintainability and allows components to evolve independently
- **Impact**: Makes the system extensible and easier to debug

### Multiple Tracking Modes
- **Decision**: Support three different gaze tracking methods
- **Why**: Different research scenarios require different precision/cost tradeoffs
- **Impact**: Makes the system accessible to users with varying hardware capabilities

### Data-Centric Design
- **Decision**: Store all events in structured formats (CSV/JSON)
- **Why**: Facilitates post-processing and integration with other tools
- **Impact**: Creates a valuable research dataset that can be analyzed with external tools

### Cross-Platform Approach
- **Decision**: Use Python with cross-platform libraries
- **Why**: Researchers work in diverse environments (Windows, macOS, Linux)
- **Impact**: Single codebase that works consistently across operating systems

### Real-Time Processing
- **Decision**: Process and visualize data as it's captured
- **Why**: Immediate feedback improves research workflow
- **Impact**: Researchers can make adjustments during sessions rather than after

## Best Practices

### 1. Separation of Concerns
- Each module has a specific responsibility (tracking, analysis, visualization)
- Clear interfaces between components minimize dependencies
- Data structures are well-defined using Python dataclasses

### 2. Error Handling and Logging
- Comprehensive logging throughout the application
- Graceful fallbacks when hardware is unavailable
- User-friendly error messages in the GUI

### 3. Configuration Flexibility
- Environment variables for global settings
- Command-line arguments for runtime configuration
- GUI settings for interactive adjustment

### 4. Thread Safety
- Proper thread synchronization for concurrent operations
- Thread-safe event handling and data collection
- Background processing to keep UI responsive

### 5. Code Documentation
- Comprehensive docstrings for all classes and methods
- Type hints for better IDE integration and code understanding
- Detailed README and supplementary documentation

### 6. Data Persistence
- Automatic saving of tracking data at regular intervals
- Multiple format support (CSV, JSON) for different use cases
- Structured filenames with timestamps for easy organization

## OS Compatibility

### macOS Considerations
- **Input Monitoring Permissions**: Detects and guides users through macOS security permissions
- **Webcam Access**: Handles camera permissions and access restrictions
- **Model File Handling**: Custom handling for downloading and extracting model files

### Windows Considerations
- **Path Handling**: Uses OS-agnostic path construction
- **Equivalent Commands**: Provides .bat scripts alongside .sh scripts
- **Administrative Access**: Minimal dependency on elevated privileges

### Linux Considerations
- **Display Server Compatibility**: Works with X11 and Wayland
- **Package Dependencies**: Clear documentation of system-level dependencies
- **Virtual Environment**: Support for isolated Python environments

### Common Approaches
- **Environment Setup**: Both bash (.sh) and batch (.bat) setup scripts
- **Portable Dependencies**: Pure Python when possible, with minimal system dependencies
- **Fallback Options**: Graceful degradation when specific features are unavailable

## Libraries and Dependencies

### Core Libraries
- **pynput**: Mouse and keyboard event monitoring
  - Why: Cross-platform, low-level input event capturing with minimal overhead
  - Usage: Primary engine for mouse tracking

- **OpenCV (cv2)**: Computer vision processing
  - Why: Industry-standard library with comprehensive image processing capabilities
  - Usage: Webcam image processing and face detection for gaze tracking

- **dlib**: Machine learning tools
  - Why: Powerful facial landmark detection with pre-trained models
  - Usage: Detecting facial features for eye tracking in webcam mode

- **numpy**: Numerical computing
  - Why: Efficient array operations and mathematical functions
  - Usage: Data processing, coordinate transformations, and statistical calculations

- **matplotlib**: Data visualization
  - Why: Comprehensive plotting library with rich features
  - Usage: Real-time plots and report visualizations

### GUI and UX Libraries
- **tkinter**: GUI framework
  - Why: Built into Python standard library, cross-platform
  - Usage: Application windows, controls, and dialogs

- **pandas**: Data analysis
  - Why: Powerful data manipulation and analysis
  - Usage: Processing event data for metrics and visualizations

### Optional Libraries
- **tobii_research**: Tobii eye tracker integration
  - Why: Official SDK for Tobii hardware
  - Usage: Precise eye tracking when hardware is available

### System Integration
- **logging**: Standard logging
  - Why: Consistent log formatting and levels
  - Usage: Tracking application behavior and debugging

- **os, sys, platform**: System interfaces
  - Why: Access to OS-specific functionality
  - Usage: Path handling, environment variables, and OS detection

## Data Flow

1. **Event Capture**: Mouse and gaze events are captured by respective trackers
2. **Synchronization**: Events are timestamped and combined in the sync tracker
3. **Processing**: Synchronized events are analyzed for patterns and metrics
4. **Visualization**: Data is rendered in real-time plots in the GUI
5. **Storage**: Events are saved to CSV/JSON files periodically and on session end
6. **Reporting**: Data is processed into comprehensive visual reports

## Future Development

### Potential Enhancements
- Machine learning integration for pattern recognition
- Heat map generation based on fixation duration
- Additional tracking hardware support
- Remote tracking and multi-user session support
- Real-time collaboration features

### Architectural Considerations
- Microservice approach for distributed processing
- Database integration for larger datasets
- Web-based visualization interface
- API for third-party integration
- Cloud synchronization for multi-device research

### Performance Optimizations
- GPU acceleration for computer vision processing
- Database indexing for faster query performance
- Memory optimization for long tracking sessions
- Batch processing for large datasets 