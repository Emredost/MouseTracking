# Mouse and Gaze Tracking API Documentation

**Copyright (c) 2025 Emre Dost (emredost1987@gmail.com). All Rights Reserved.**

This API documentation is for developers who wish to extend the functionality of the Mouse and Gaze Tracking system. This software was developed for UX research, usability testing, cognitive load assessment, accessibility research, and human-computer interaction (HCI) studies at ESTIA University of Technology, France.

## Table of Contents

1. [Overview](#overview)
2. [Core Modules](#core-modules)
    - [MouseTracker](#mousetracker)
    - [GazeTracker](#gazetracker)
    - [SyncTracker](#synctracker)
3. [Data Structures](#data-structures)
    - [MouseEvent](#mouseevent)
    - [GazeEvent](#gazeevent)
    - [SyncEvent](#syncevent)
4. [Analysis Module](#analysis-module)
5. [GUI Integration](#gui-integration)
6. [Extension Points](#extension-points)
7. [Best Practices](#best-practices)
8. [Licensing](#licensing)

## Overview

The Mouse and Gaze Tracking system is designed with a modular architecture that separates core tracking functionality from visualization and analysis. Each module is designed to work independently or as part of the integrated system.

The system architecture follows this pattern:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  MouseTracker   │     │   GazeTracker   │     │  MouseAnalytics │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────┬───────────┘                       │
                     │                                   │
               ┌─────▼─────┐                             │
               │SyncTracker◄─────────────────────────────┘
               └─────┬─────┘
                     │
               ┌─────▼─────┐
               │    GUI    │
               └───────────┘
```

## Core Modules

### MouseTracker

The `MouseTracker` class is responsible for capturing mouse events and storing them for analysis.

#### Key Methods

```python
class MouseTracker:
    def __init__(self, output_dir: str = DEFAULT_DATA_DIR):
        """Initialize the mouse tracker
        
        Args:
            output_dir: Directory to save tracking data
        """
        
    def start(self) -> bool:
        """Start mouse tracking
        
        Returns:
            bool: True if tracking started successfully
        """
        
    def stop(self) -> None:
        """Stop mouse tracking and save collected data"""
        
    def save_data(self) -> Tuple[str, str]:
        """Save tracking data to files
        
        Returns:
            Tuple[str, str]: Paths to the created CSV and JSON files
        """
        
    def generate_heatmap(self, grid_size: int = 50) -> str:
        """Generate a heatmap visualization of mouse positions
        
        Args:
            grid_size: Resolution of the heatmap grid
            
        Returns:
            str: Path to the saved figure
        """
```

#### Events

The `MouseTracker` captures three types of events:
- `move`: Mouse movement with x,y coordinates
- `click`: Mouse button clicks (left, right, middle) with pressed state
- `scroll`: Scroll wheel events with dx,dy values

### GazeTracker

The `GazeTracker` class captures eye gaze data through various tracking methods.

#### Key Methods

```python
class GazeTracker:
    def __init__(self, output_dir: str = DEFAULT_DATA_DIR, 
                 mode: str = GAZE_TRACKER_MODE,
                 screen_resolution: Tuple[int, int] = None):
        """Initialize the gaze tracker
        
        Args:
            output_dir: Directory to save tracking data
            mode: Tracking method ('webcam', 'tobii', or 'dummy')
            screen_resolution: Screen dimensions in pixels
        """
        
    def calibrate(self, calibration_points: List[Tuple[float, float]] = None) -> bool:
        """Calibrate the gaze tracker
        
        Args:
            calibration_points: Screen positions for calibration (normalized 0-1)
            
        Returns:
            bool: True if calibration was successful
        """
        
    def start(self) -> bool:
        """Start gaze tracking
        
        Returns:
            bool: True if tracking started successfully
        """
        
    def stop(self) -> None:
        """Stop gaze tracking and save collected data"""
        
    def save_data(self) -> None:
        """Save gaze tracking data to files"""
```

#### Events

The `GazeTracker` captures three types of events:
- `fixation`: When the eye focuses on a point
- `saccade`: Rapid eye movement between fixations
- `blink`: Eye blink events

### SyncTracker

The `SyncTracker` class synchronizes and correlates mouse and gaze data.

#### Key Methods

```python
class SyncTracker:
    def __init__(self, output_dir: str = DEFAULT_DATA_DIR,
                 gaze_mode: str = GAZE_TRACKER_MODE,
                 screen_resolution: Tuple[int, int] = None):
        """Initialize the synchronized tracker
        
        Args:
            output_dir: Directory to save tracking data
            gaze_mode: Gaze tracking mode
            screen_resolution: Screen dimensions in pixels
        """
        
    def start(self) -> bool:
        """Start synchronized tracking
        
        Returns:
            bool: True if tracking started successfully
        """
        
    def stop(self) -> None:
        """Stop synchronized tracking"""
        
    def save_data(self) -> Tuple[str, str]:
        """Save synchronized tracking data to files
        
        Returns:
            Tuple[str, str]: Paths to the created CSV and JSON files
        """
        
    def generate_heatmap_comparison(self) -> str:
        """Generate a comparison heatmap of mouse and gaze positions
        
        Returns:
            str: Path to the saved figure
        """
        
    def generate_distance_plot(self) -> str:
        """Generate a plot of mouse-gaze distance over time
        
        Returns:
            str: Path to the saved figure
        """
        
    def generate_trajectory_comparison(self) -> str:
        """Generate a comparison of mouse and gaze trajectories
        
        Returns:
            str: Path to the saved figure
        """
        
    def generate_report(self) -> str:
        """Generate a comprehensive HTML report
        
        Returns:
            str: Path to the saved HTML report
        """
```

## Data Structures

### MouseEvent

```python
@dataclass
class MouseEvent:
    """Data class for mouse events"""
    timestamp: float         # When the event happened (seconds since epoch)
    event_type: str          # 'move', 'click', or 'scroll'
    x: int = 0               # X-coordinate on screen
    y: int = 0               # Y-coordinate on screen
    button: Optional[str] = None    # For clicks: 'left', 'right', 'middle'
    pressed: Optional[bool] = None  # For clicks: True for press, False for release
    dx: Optional[int] = None        # For scrolls: horizontal scroll amount
    dy: Optional[int] = None        # For scrolls: vertical scroll amount
```

### GazeEvent

```python
@dataclass
class GazeEvent:
    """Data structure for storing gaze events"""
    timestamp: float         # When the event happened (seconds since epoch)
    event_type: str          # 'fixation', 'saccade', or 'blink'
    x: float = 0.0           # X-coordinate on screen (normalized 0-1)
    y: float = 0.0           # Y-coordinate on screen (normalized 0-1)
    duration: Optional[float] = None  # How long the fixation lasted (seconds)
    pupil_size: Optional[float] = None  # Size of the pupil (millimeters)
    confidence: float = 1.0  # How reliable the measurement is (0-1)
    screen_x: Optional[int] = None  # Actual pixel X-coordinate on screen
    screen_y: Optional[int] = None  # Actual pixel Y-coordinate on screen
```

### SyncEvent

```python
@dataclass
class SyncEvent:
    """Data class for synchronized mouse and gaze events"""
    timestamp: float
    event_type: str  # 'mouse_move', 'mouse_click', 'mouse_scroll', 'gaze_fixation', 'gaze_saccade', 'gaze_blink'
    mouse_x: Optional[int] = None
    mouse_y: Optional[int] = None
    gaze_x: Optional[float] = None
    gaze_y: Optional[float] = None
    gaze_screen_x: Optional[int] = None  
    gaze_screen_y: Optional[int] = None
    mouse_button: Optional[str] = None
    mouse_pressed: Optional[bool] = None
    mouse_dx: Optional[int] = None
    mouse_dy: Optional[int] = None
    gaze_duration: Optional[float] = None
    gaze_pupil_size: Optional[float] = None
    gaze_confidence: Optional[float] = None
    distance: Optional[float] = None  # Distance between mouse and gaze in pixels
    normalized_distance: Optional[float] = None  # Normalized distance (0-1)
```

## Analysis Module

The `MouseAnalytics` class provides various methods for analyzing tracking data:

```python
class MouseAnalytics:
    def __init__(self, data_dir: str = "mouse_data"):
        """Initialize the analytics engine"""
        
    def load_data(self, file_pattern: str = "*.json") -> None:
        """Load data from JSON files in the data directory"""
        
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate key metrics from the tracking data"""
        
    def generate_trajectory_plot(self, output_file: str = None) -> None:
        """Generate a trajectory plot of mouse movements"""
        
    def generate_heatmap(self, grid_size: int = 50, output_file: str = None) -> None:
        """Generate a heatmap of mouse positions"""
        
    def generate_speed_plot(self, output_file: str = None) -> None:
        """Generate a plot of mouse movement speed over time"""
        
    def generate_click_distribution(self, output_file: str = None) -> None:
        """Generate a plot showing the distribution of mouse clicks"""
        
    def generate_report(self, output_file: str = None) -> None:
        """Generate a comprehensive HTML report of mouse activity"""
```

## GUI Integration

The system includes two GUI interfaces:

1. `MouseTrackerGUI`: Basic interface for mouse tracking
2. `SyncTrackerGUI`: Advanced interface for synchronized mouse and gaze tracking

These classes handle the visualization and user interaction. They use the `RealTimeTrajectoryPlot` and `DistancePlot` classes for real-time visualization.

## Extension Points

When extending the system, focus on these key extension points:

### 1. New Tracking Methods

To add a new gaze tracking method to `GazeTracker`:

1. Create a new initialization method in the `GazeTracker` class:
```python
def _init_new_tracker_method(self):
    # Initialize your tracking method
    pass
```

2. Modify the `__init__` method to include your new tracking method

### 2. Custom Visualizations

To create new visualizations:

1. Add a new generation method to `SyncTracker` or `MouseAnalytics`
2. Integrate it with the report generation

### 3. New Analysis Metrics

To add new analysis metrics:

1. Extend the `calculate_metrics` method in `MouseAnalytics`
2. Add the new metrics to the report template

### 4. Custom Event Types

To add custom event types:

1. Modify the appropriate data class (`MouseEvent`, `GazeEvent`, or `SyncEvent`)
2. Update the event handling in the corresponding tracker class

## Best Practices

1. **Maintain Thread Safety**: All tracker classes use threading for data collection. Always use locks when accessing shared data.

2. **Error Handling**: Use try-except blocks for hardware interactions and fallback to simpler modes when necessary.

3. **Resource Management**: Always release resources (e.g., camera) in the `stop` method.

4. **Data Storage**: Follow the existing patterns for data storage to maintain compatibility.

5. **Visualization**: Use matplotlib with consistent styling for all visualizations.

## Licensing

This code is released under a Research and Academic License. To use, modify, or extend this code:

1. Permission to use the Software for academic and research purposes may be granted upon request by emailing emredost1987@gmail.com
2. The Software may be used for non-commercial research and educational purposes only
3. You must acknowledge the original author (Emre Dost) and ESTIA University of Technology in any publications or presentations that result from the use of this Software
4. Redistributions of the Software must retain the license notice and acknowledgment
5. For any commercial use, explicit written permission is required from the copyright holder

---

*This documentation is for academic and research purposes. For permission to use or distribute, please contact Emre Dost at emredost1987@gmail.com.* 