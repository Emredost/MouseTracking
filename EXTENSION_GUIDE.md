# Extension Guide: Adding New Tracking Methods

**Copyright (c) 2025 Emre Dost (emredost1987@gmail.com). All Rights Reserved.**

This guide demonstrates how to extend the Mouse and Gaze Tracking system with new tracking methods. This software was developed for UX research, usability testing, cognitive load assessment, accessibility research, and human-computer interaction (HCI) studies at ESTIA University of Technology, France.

## Overview

The system is designed to be extensible in several ways, with adding new tracking methods being one of the most common extensions. This guide will walk through the process of adding a new gaze tracking method to the `GazeTracker` class.

## Step 1: Understand the Current Architecture

The `GazeTracker` class currently supports three tracking methods:
- `webcam`: Uses computer vision to track gaze
- `tobii`: Interfaces with Tobii eye tracking hardware
- `dummy`: Generates synthetic data for testing

Each tracking method is initialized in a separate method:
- `_init_webcam_tracker()`
- `_init_tobii_tracker()`
- `_init_dummy_tracker()`

## Step 2: Define Your New Tracking Method

Let's implement a hypothetical new tracking method called "neural" that uses a neural network model for more accurate gaze prediction. Here's how you would add it:

### 2.1 Add the Initialization Method

```python
def _init_neural_tracker(self):
    """
    Initialize neural network-based gaze tracker
    
    This uses a pre-trained neural network model to predict
    gaze points with higher accuracy than the basic webcam tracker
    """
    try:
        # Check for required libraries
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        
        # Path to the model
        model_path = os.path.join(os.path.dirname(__file__), "models", "neural_gaze_model.h5")
        
        # Check if model exists
        if not os.path.exists(model_path):
            model_dir = os.path.dirname(model_path)
            os.makedirs(model_dir, exist_ok=True)
            logger.error(f"Neural gaze model not found at {model_path}. "
                         f"Please download it and place it in {model_dir}")
            self.mode = 'dummy'
            self._init_dummy_tracker()
            return
        
        # Load the model
        self.neural_model = load_model(model_path)
        logger.info("Neural model loaded successfully")
        
        # Initialize webcam (we'll still need camera input)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logger.error("Could not open webcam. Falling back to dummy mode.")
            self.mode = 'dummy'
            self._init_dummy_tracker()
            return
        
        # Set up threading like the webcam tracker
        self.frame_ready = threading.Event()
        self.current_frame = None
        self.stop_requested = False
        
        # Set calibration status
        self.calibrated = False
        
        logger.info("Neural gaze tracker initialized")
    except Exception as e:
        logger.error(f"Error initializing neural tracker: {e}")
        logger.error("Falling back to dummy tracker mode")
        self.mode = 'dummy'
        self._init_dummy_tracker()
```

### 2.2 Update the `__init__` Method to Include Your New Mode

Modify the `__init__` method in `GazeTracker` to include your new tracking method:

```python
def __init__(self, output_dir: str = DEFAULT_DATA_DIR, 
             mode: str = GAZE_TRACKER_MODE,
             screen_resolution: Tuple[int, int] = None):
    # ... existing initialization code ...
    
    # Initialize tracker based on mode
    if self.mode == 'webcam':
        self._init_webcam_tracker()
    elif self.mode == 'tobii':
        self._init_tobii_tracker()
    elif self.mode == 'neural':  # Add this new condition
        self._init_neural_tracker()
    else:  # 'dummy' or fallback
        self._init_dummy_tracker()
    
    # ... rest of initialization code ...
```

### 2.3 Implement the Processing Method

Add a new method to process frames with your neural network model:

```python
def _process_neural_frames(self):
    """Process webcam frames using the neural network model"""
    while not self.stop_requested:
        if not self.cap.isOpened():
            logger.error("Webcam disconnected")
            break
            
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to capture frame")
            continue
        
        # Prepare the frame for the neural model
        processed_frame = cv2.resize(frame, (224, 224))  # Example size
        processed_frame = processed_frame / 255.0  # Normalize
        processed_frame = np.expand_dims(processed_frame, axis=0)  # Add batch dimension
        
        # Get prediction from the model
        prediction = self.neural_model.predict(processed_frame, verbose=0)
        
        # Extract normalized gaze coordinates (example format)
        norm_x, norm_y = prediction[0]
        
        # Convert to screen coordinates
        screen_x = int(norm_x * self.screen_resolution[0])
        screen_y = int(norm_y * self.screen_resolution[1])
        
        # Create and store gaze event
        with self.lock:
            self.events.append(GazeEvent(
                timestamp=time.time(),
                event_type='fixation',
                x=norm_x,
                y=norm_y,
                screen_x=screen_x,
                screen_y=screen_y,
                confidence=0.9  # Assume high confidence from neural network
            ))
        
        # Update current frame for calibration purposes
        self.current_frame = frame.copy()
        self.frame_ready.set()
        
        # Sleep to control frame rate
        time.sleep(0.01)
```

### 2.4 Update the Start Method

Modify the `start()` method to use your new processing thread:

```python
def start(self) -> bool:
    # ... existing code ...
    
    if self.mode == 'webcam':
        # Start the frame processing thread
        self.stop_requested = False
        self.processing_thread = threading.Thread(target=self._process_webcam_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    elif self.mode == 'neural':  # Add this condition
        # Start the neural processing thread
        self.stop_requested = False
        self.processing_thread = threading.Thread(target=self._process_neural_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    elif self.mode == 'tobii':
        # ... existing tobii code ...
    elif self.mode == 'dummy':
        # ... existing dummy code ...
    
    return True
```

### 2.5 Update the Stop Method

Make sure your new resources are properly released:

```python
def stop(self) -> None:
    # ... existing code ...
    
    if self.mode == 'webcam' or self.mode == 'neural':  # Add neural here
        # Stop the processing thread
        self.stop_requested = True
        if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        
        # Release the webcam
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
    
    # ... rest of the code ...
```

## Step 3: Update Environment Variables and Constants

Add your new tracking mode to the environment variable options:

```python
# In the appropriate configuration section:
GAZE_TRACKER_MODE = os.environ.get('GAZE_TRACKER_MODE', 'webcam').lower()  # 'webcam', 'tobii', 'neural', or 'dummy'
```

## Step 4: Update the UI Options

If you're using the GUI, update the gaze mode dropdown to include your new option:

```python
# In sync_tracker_gui.py, find the combobox for gaze mode:
gaze_mode_combo = ttk.Combobox(control_frame, textvariable=self.gaze_mode_var, 
                              values=["webcam", "tobii", "neural", "dummy"],
                              width=10, state="readonly")
```

## Step 5: Add Documentation

Document your new tracking method:

```python
def _init_neural_tracker(self):
    """
    Initialize neural network-based gaze tracker
    
    This method initializes a gaze tracker that uses a pre-trained 
    neural network model to predict gaze points from webcam images.
    It offers improved accuracy compared to the geometric approach
    used by the basic webcam tracker.
    
    Requirements:
    - TensorFlow 2.x
    - The neural_gaze_model.h5 file in the models directory
    - A working webcam
    
    If initialization fails, it falls back to the dummy tracker.
    """
    # ... implementation ...
```

## Step 6: Testing

Test your implementation thoroughly:

1. Test initialization
2. Test with and without required models/hardware
3. Test calibration if applicable
4. Test tracking accuracy
5. Test resource cleanup

## Common Pitfalls

1. **Thread Safety**: Always use `self.lock` when modifying `self.events`
2. **Resource Management**: Always clean up resources like webcams
3. **Graceful Fallbacks**: Fall back to the dummy tracker when things go wrong
4. **Error Handling**: Use try-except blocks for all external dependencies
5. **Performance**: Be mindful of CPU/GPU usage in real-time tracking

## Advanced: Adding Calibration Support

If your new tracking method requires calibration, you'll need to update the `calibrate()` method:

```python
def calibrate(self, calibration_points: List[Tuple[float, float]] = None) -> bool:
    # ... existing code ...
    
    elif self.mode == 'neural':
        # Neural network-specific calibration
        if not hasattr(self, 'neural_model'):
            logger.error("Neural model not initialized")
            return False
            
        # Implement your calibration logic here
        # For example, collecting eye images at known gaze points
        # and fine-tuning the model
        
        self.calibrated = True
        logger.info("Neural tracker calibration completed")
        return True
    
    # ... rest of the method ...
```

## Conclusion

By following this guide, you can add new tracking methods to the existing system in a way that maintains compatibility with the rest of the codebase. Remember to always handle errors gracefully and provide fallbacks when things go wrong.

**Note**: This extension guide is part of a software developed for academic and research purposes. For permission to use or extend the software, please contact Emre Dost at emredost1987@gmail.com. 