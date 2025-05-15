# Mouse and Gaze Tracking API Quick Reference

**Copyright (c) 2025 Emre Dost (emredost1987@gmail.com). All Rights Reserved.**

This quick reference guide provides an overview of the key classes and methods in the Mouse and Gaze Tracking API. For detailed documentation, see `API_DOCUMENTATION.md`.

## Core Classes

### MouseTracker
```python
tracker = MouseTracker(output_dir="mouse_data")
tracker.start()                        # Start tracking
tracker.stop()                         # Stop tracking
tracker.save_data()                    # Save data manually
tracker.events                         # Access collected events
```

### GazeTracker
```python
tracker = GazeTracker(output_dir="mouse_data", mode="webcam")
tracker.calibrate()                    # Calibrate the tracker
tracker.start()                        # Start tracking
tracker.stop()                         # Stop tracking
tracker.save_data()                    # Save data manually
tracker.events                         # Access collected events
```

### SyncTracker
```python
tracker = SyncTracker(output_dir="mouse_data", gaze_mode="webcam")
tracker.start()                        # Start synchronized tracking
tracker.stop()                         # Stop tracking
tracker.save_data()                    # Save data manually
tracker.sync_events                    # Access synchronized events
tracker.generate_report()              # Generate HTML report
```

## Data Classes

### MouseEvent
```python
MouseEvent(
    timestamp=time.time(),           # Current time
    event_type="move",               # "move", "click", or "scroll"
    x=100, y=200,                    # Coordinates
    button="left",                   # For clicks: "left", "right", "middle"
    pressed=True,                    # For clicks: press=True, release=False
    dx=0, dy=0                       # For scrolls: scroll amounts
)
```

### GazeEvent
```python
GazeEvent(
    timestamp=time.time(),           # Current time
    event_type="fixation",           # "fixation", "saccade", or "blink"
    x=0.5, y=0.5,                    # Normalized coordinates (0-1)
    screen_x=960, screen_y=540,      # Screen pixel coordinates
    duration=0.2,                    # Fixation duration in seconds
    pupil_size=3.5,                  # Pupil diameter in mm
    confidence=0.9                   # Measurement confidence (0-1)
)
```

### SyncEvent
```python
SyncEvent(
    timestamp=time.time(),           # Current time
    event_type="mouse_move",         # Event type prefix indicates source
    mouse_x=100, mouse_y=100,        # Mouse coordinates
    gaze_x=0.5, gaze_y=0.5,          # Normalized gaze coordinates
    gaze_screen_x=960, gaze_screen_y=540,  # Screen gaze coordinates
    distance=150,                    # Distance between mouse and gaze
    normalized_distance=0.1          # Distance normalized by screen diagonal
)
```

## Analysis

### MouseAnalytics
```python
analytics = MouseAnalytics(data_dir="mouse_data")
analytics.load_data()                  # Load data from files
metrics = analytics.calculate_metrics()  # Calculate metrics
analytics.generate_trajectory_plot()   # Generate plots
analytics.generate_heatmap()
analytics.generate_report()            # Generate HTML report
```

## GUI Integration

### MouseTrackerGUI
```python
root = tk.Tk()
app = MouseTrackerGUI(root)            # Create GUI
root.mainloop()                        # Run GUI
```

### SyncTrackerGUI
```python
root = tk.Tk()
app = SyncTrackerGUI(root)             # Create GUI
root.mainloop()                        # Run GUI
```

## Common Patterns

### Recording Session
```python
# Initialize tracker
tracker = MouseTracker(output_dir="mouse_data")

# Start tracking
tracker.start()

# Run for a specific duration
time.sleep(60)  # 60 seconds

# Stop tracking (automatically saves data)
tracker.stop()
```

### Accessing Events in Real-time
```python
# Initialize tracker
tracker = MouseTracker(output_dir="mouse_data")
tracker.start()

# Process events as they come in
last_idx = 0
while True:
    # Get new events
    with tracker.lock:
        new_events = tracker.events[last_idx:]
        last_idx = len(tracker.events)
    
    # Process new events
    for event in new_events:
        print(f"Event: {event.event_type} at ({event.x}, {event.y})")
    
    time.sleep(0.1)  # Check every 100ms
```

### Generating Custom Visualizations
```python
# Load data
analytics = MouseAnalytics(data_dir="mouse_data")
analytics.load_data()

# Create custom plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot click locations
click_data = analytics.click_data
ax.scatter(click_data['x'], click_data['y'], alpha=0.7, s=30)
ax.set_title('Click Locations')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.invert_yaxis()  # Invert y-axis to match screen coordinates

# Save plot
plt.savefig('custom_plot.png')
```

## Extension Example: Custom Event Processing

```python
# Extend SyncTracker with custom processing
class EnhancedSyncTracker(SyncTracker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hover_duration = {}  # Track hover duration by screen region
    
    def _sync_events_thread(self):
        # Call the parent method
        super()._sync_events_thread()
        
        # Add custom processing for each event
        for event in self.sync_events:
            if event.event_type == 'mouse_move':
                # Calculate screen region (divide screen into 3x3 grid)
                x_region = event.mouse_x // (self.screen_resolution[0] // 3)
                y_region = event.mouse_y // (self.screen_resolution[1] // 3)
                region = (x_region, y_region)
                
                # Update hover time for this region
                if region in self.hover_duration:
                    self.hover_duration[region] += sync_interval
                else:
                    self.hover_duration[region] = sync_interval
```

---

**Note**: This software was developed for UX research, usability testing, cognitive load assessment, accessibility research, and human-computer interaction (HCI) studies at ESTIA University of Technology, France. For permission to use for academic and research purposes, please contact Emre Dost at emredost1987@gmail.com. 