#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ======================================================
# Gaze Tracker - Eye movement tracking system
# 
# This program records all eye gaze activities including:
# - Eye fixations
# - Saccades (rapid eye movements)
# - Blinks
# 
# Data is automatically saved and can be visualized
# as heatmaps to show gaze patterns
# ======================================================

import time
import datetime
import os
import threading
import logging
import numpy as np
import cv2
import dlib # For face detection and landmark prediction
import json
import csv
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional, Union

# ====== CONFIGURATION ======
# Default settings that can be overridden by environment variables
# This can be changed by setting the MOUSE_TRACKER_DATA_DIR environment variable
DEFAULT_DATA_DIR = os.environ.get('MOUSE_TRACKER_DATA_DIR', os.path.join(os.getcwd(), "mouse_data"))
DEBUG_MODE = os.environ.get('MOUSE_TRACKER_DEBUG', 'false').lower() == 'true'
GAZE_TRACKER_MODE = os.environ.get('GAZE_TRACKER_MODE', 'webcam').lower()  # 'webcam', 'tobii', or 'dummy'

# Setting up logging to track what's happening in the program
log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(os.path.join("logs", "gaze_tracker.log")), logging.StreamHandler()]
)
logger = logging.getLogger("GazeTracker")

@dataclass
class GazeEvent:
    """
    Data structure for storing gaze events
    Each event captures what the eye did at a specific moment in time
    """
    timestamp: float         # When the event happened (seconds since epoch)
    event_type: str          # What kind of event: 'fixation', 'saccade', or 'blink'
    x: float = 0.0           # X-coordinate on screen (normalized 0-1)
    y: float = 0.0           # Y-coordinate on screen (normalized 0-1)
    duration: Optional[float] = None  # How long the fixation lasted (seconds)
    pupil_size: Optional[float] = None  # Size of the pupil (millimeters)
    confidence: float = 1.0  # How reliable the measurement is (0-1)
    screen_x: Optional[int] = None  # Actual pixel X-coordinate on screen
    screen_y: Optional[int] = None  # Actual pixel Y-coordinate on screen

class GazeTracker:
    """
    The main tracking system that records and analyzes eye movements
    """
    
    def __init__(self, output_dir: str = DEFAULT_DATA_DIR, 
                 mode: str = GAZE_TRACKER_MODE,
                 screen_resolution: Tuple[int, int] = None):
        """
        Initialize the tracker with basic settings
        
        Args:
            output_dir: Where to save the collected data
            mode: Which tracking method to use ('webcam', 'tobii', or 'dummy')
            screen_resolution: The size of the screen in pixels (width, height)
        """
        # Storage for all gaze events
        self.events: List[GazeEvent] = [] # List to store gaze events
        self.running = False # Flag to control the tracking loop so we can stop it gracefully
        self.output_dir = output_dir # Directory to save data
        self.mode = mode # Tracking mode: 'webcam', 'tobii', or 'dummy'
        self.tracker = None 
        self.lock = threading.Lock()  # Prevents data corruption when multiple threads access data
        self.calibrated = False # Flag to check if the tracker is calibrated
        
        # Get screen resolution for gaze mapping 
        if screen_resolution is None:
            try:
                import tkinter as tk
                root = tk.Tk()
                screen_width = root.winfo_screenwidth() # Width in pixels
                screen_height = root.winfo_screenheight() # Height in pixels
                root.destroy() # Close the Tkinter window and set resolution
                self.screen_resolution = (screen_width, screen_height)
            except:
                # Default resolution if we can't detect
                self.screen_resolution = (1920, 1080)
                logger.warning("Could not detect screen resolution. Using default: 1920x1080")
        else:
            self.screen_resolution = screen_resolution
        
        # Making sure we have a place to save our data
        os.makedirs(output_dir, exist_ok=True)
        
        # Initializing tracker based on mode
        if self.mode == 'webcam':
            self._init_webcam_tracker()
        elif self.mode == 'tobii':
            self._init_tobii_tracker()
        else:  # 'dummy' or fallback
            self._init_dummy_tracker()
            
        # Tracking statistics
        self.total_fixation_duration = 0.0  # Total time spent on fixations
        self.fixation_count = 0             # Number of fixation events
        self.saccade_count = 0              # Number of saccade events
        self.blink_count = 0                # Number of blink events
        self.start_time = 0                 # When tracking began
        
        logger.info(f"GazeTracker initialized in {self.mode} mode")
        logger.debug(f"Using output directory: {self.output_dir}")
        logger.debug(f"Screen resolution: {self.screen_resolution}")
    
    def _init_webcam_tracker(self):
        """
        Initialize webcam-based gaze tracker using computer vision
        
        This uses the webcam to detect the user's face and eyes,
        then calculates where they are looking on the screen
        """
        try:
            # Check if we have necessary libraries and models
            self.detector = dlib.get_frontal_face_detector()
            
            # Path to face landmark predictor model that i found useful
            # This is a pre-trained model for facial landmark detection
            # You can download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
            model_path = os.path.join(os.path.dirname(__file__), "models", "shape_predictor_68_face_landmarks.dat")
            
            # If model doesn't exist, prompt user to download it
            if not os.path.exists(model_path):
                model_dir = os.path.dirname(model_path)
                os.makedirs(model_dir, exist_ok=True)
                logger.error(f"Face landmark model not found. Please download it from: "
                             f"http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2, "
                             f"extract it and place it in {model_dir}")
                self.mode = 'dummy'
                self._init_dummy_tracker()
                return
            
            self.predictor = dlib.shape_predictor(model_path)
            
            # Initialize webcam
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logger.error("Could not open webcam. Falling back to dummy mode.")
                self.mode = 'dummy'
                self._init_dummy_tracker()
                return
            
            # Set up thread for processing webcam frames
            self.frame_ready = threading.Event() # Event to signal when a frame is ready
            self.current_frame = None # Current frame for calibration
            self.stop_requested = False # Flag to stop the thread 
            
            # Calibration data
            self.calibration_data = [] # List to store calibration points
            self.calibration_points = [] # List to store calibration points
            self.calibration_matrix = np.eye(3)  # Identity matrix as default
            
            # Eye appearance model
            self.eye_left = None # Left eye appearance model so we can track it 
            self.eye_right = None
            
            logger.info("Webcam-based gaze tracker initialized")
        except Exception as e:
            logger.error(f"Error initializing webcam tracker: {e}")
            logger.error("Falling back to dummy tracker mode")
            self.mode = 'dummy'
            self._init_dummy_tracker()
    
    def _init_tobii_tracker(self):
        """
        Initialize Tobii eye tracker hardware
        
        This connects to a Tobii eye tracker device if available
        and sets up the data collection process
        """
        try:
            # Attempt to import tobii_research
            import tobii_research as tr # This is the Tobii Research SDK for Python 
            
            # Find connected eye trackers
            eye_trackers = tr.find_all_eyetrackers() # List of all connected eye trackers
            
            if len(eye_trackers) == 0:
                logger.error("No Tobii eye trackers found. Falling back to dummy mode.")
                self.mode = 'dummy'
                self._init_dummy_tracker()
                return
            
            # Use the first available tracker
            self.tracker = eye_trackers[0]
            # Set the screen resolution based on the tracker 
            logger.info(f"Connected to Tobii eye tracker: {self.tracker.model} (S/N: {self.tracker.serial_number})")
            # Create callback for gaze data
            def gaze_data_callback(gaze_data):
                # Process and store gaze data
                if gaze_data['left_gaze_point_validity'] or gaze_data['right_gaze_point_validity']:
                    # Use the valid eye, or average if both are valid
                    x, y = 0.0, 0.0 # Normalized coordinates (0-1)
                    valid_eyes = 0 # Count valid eyes
                    
                    if gaze_data['left_gaze_point_validity']:
                        x += gaze_data['left_gaze_point_on_display_area'][0]
                        y += gaze_data['left_gaze_point_on_display_area'][1]
                        valid_eyes += 1
                    
                    if gaze_data['right_gaze_point_validity']:
                        x += gaze_data['right_gaze_point_on_display_area'][0]
                        y += gaze_data['right_gaze_point_on_display_area'][1]
                        valid_eyes += 1
                    
                    x /= valid_eyes
                    y /= valid_eyes
                    
                    # Convert normalized coordinates to screen coordinates
                    screen_x = int(x * self.screen_resolution[0])
                    screen_y = int(y * self.screen_resolution[1])
                    
                    # Get pupil size (average of both eyes if available)
                    pupil_size = 0.0
                    if gaze_data['left_pupil_validity']:
                        pupil_size += gaze_data['left_pupil_diameter']
                        valid_pupils = 1
                    else:
                        valid_pupils = 0
                    
                    if gaze_data['right_pupil_validity']:
                        pupil_size += gaze_data['right_pupil_diameter']
                        valid_pupils += 1
                    
                    if valid_pupils > 0:
                        pupil_size /= valid_pupils
                    else:
                        pupil_size = None
                    
                    # Determine if this is a fixation (simplified - in real applications,
                    # you'd use a fixation detection algorithm)
                    # For now, we'll just assume all valid gaze points are fixations
                    event = GazeEvent(
                        timestamp=time.time(),
                        event_type='fixation',
                        x=x,
                        y=y,
                        screen_x=screen_x,
                        screen_y=screen_y,
                        pupil_size=pupil_size,
                        confidence=1.0 if valid_eyes == 2 else 0.5
                    )
                    
                    with self.lock:
                        self.events.append(event)
            
            # Store the callback for later unsubscription
            self.gaze_callback = gaze_data_callback
            
            # Set the calibrated flag
            self.calibrated = True
            
            logger.info("Tobii eye tracker initialized")
        except ImportError:
            logger.error("Tobii Research SDK not installed. Falling back to dummy mode.")
            self.mode = 'dummy'
            self._init_dummy_tracker()
        except Exception as e:
            logger.error(f"Error initializing Tobii tracker: {e}")
            logger.error("Falling back to dummy tracker mode")
            self.mode = 'dummy'
            self._init_dummy_tracker()
    
    def _init_dummy_tracker(self):
        """
        Initialize a dummy tracker that simulates gaze data
        
        This is used when no real eye tracking is available,
        creating realistic synthetic data for testing
        """
        self.mode = 'dummy'
        # There's no actual initialization needed for the dummy mode,
        # as it will just generate synthetic data
        logger.info("Dummy gaze tracker initialized")
        self.calibrated = True  # Dummy tracker is always "calibrated"
    
    def _process_webcam_frames(self):
        """
        Process webcam frames to track gaze in a separate thread
        
        This analyzes each video frame to detect eyes and estimate gaze direction
        """
        last_blink_time = 0
        minimum_blink_interval = 1.0  # Minimum time between blinks in seconds 
        # so that we won't count multiple blinks to avoid counting multiple blinks in quick succession 
        
        while not self.stop_requested:
            if not self.cap.isOpened():
                logger.error("Webcam disconnected")
                break
                
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to capture frame")
                continue
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.detector(gray)
            
            if len(faces) > 0:
                # Process the first face found
                face = faces[0]
                
                # Get facial landmarks
                landmarks = self.predictor(gray, face)
                
                # Extract eye regions
                left_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
                right_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
                
                # Calculate eye aspect ratio to detect blinks
                def eye_aspect_ratio(eye_points):
                    # Compute the distances
                    vertical_1 = np.linalg.norm(eye_points[1] - eye_points[5])
                    vertical_2 = np.linalg.norm(eye_points[2] - eye_points[4])
                    horizontal = np.linalg.norm(eye_points[0] - eye_points[3])
                    
                    # Calculate eye aspect ratio
                    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
                    return ear
                
                left_ear = eye_aspect_ratio(left_eye_points)
                right_ear = eye_aspect_ratio(right_eye_points)
                
                # Average EAR
                ear = (left_ear + right_ear) / 2.0
                
                # Blink detection
                current_time = time.time()
                if ear < 0.2 and current_time - last_blink_time > minimum_blink_interval:
                    # This is a blink
                    with self.lock:
                        self.events.append(GazeEvent(
                            timestamp=current_time,
                            event_type='blink',
                            confidence=1.0
                        ))
                        
                        self.blink_count += 1
                        last_blink_time = current_time
                
                # Pupil detection and gaze estimation (simplified)
                left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
                right_eye_center = np.mean(right_eye_points, axis=0).astype(int)
                
                # Extract eye regions for pupil detection
                left_eye_region = frame[
                    max(0, left_eye_center[1] - 10):min(frame.shape[0], left_eye_center[1] + 10),
                    max(0, left_eye_center[0] - 15):min(frame.shape[1], left_eye_center[0] + 15)
                ]
                
                right_eye_region = frame[
                    max(0, right_eye_center[1] - 10):min(frame.shape[0], right_eye_center[1] + 10),
                    max(0, right_eye_center[0] - 15):min(frame.shape[1], right_eye_center[0] + 15)
                ]
                
                # Skip if eye regions are too small
                if left_eye_region.size > 0 and right_eye_region.size > 0:
                    # Convert to grayscale
                    left_eye_gray = cv2.cvtColor(left_eye_region, cv2.COLOR_BGR2GRAY)
                    right_eye_gray = cv2.cvtColor(right_eye_region, cv2.COLOR_BGR2GRAY)
                    
                    # Apply Gaussian blur
                    left_eye_gray = cv2.GaussianBlur(left_eye_gray, (5, 5), 0)
                    right_eye_gray = cv2.GaussianBlur(right_eye_gray, (5, 5), 0)
                    
                    # Threshold to find dark regions (pupils)
                    _, left_threshold = cv2.threshold(left_eye_gray, 50, 255, cv2.THRESH_BINARY_INV)
                    _, right_threshold = cv2.threshold(right_eye_gray, 50, 255, cv2.THRESH_BINARY_INV)
                    
                    # Find contours
                    left_contours, _ = cv2.findContours(left_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    right_contours, _ = cv2.findContours(right_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # If contours found, find the largest (likely the pupil)
                    left_pupil = None
                    if left_contours:
                        left_pupil = max(left_contours, key=cv2.contourArea)
                        
                    right_pupil = None
                    if right_contours:
                        right_pupil = max(right_contours, key=cv2.contourArea)
                    
                    if left_pupil is not None and right_pupil is not None:
                        # Calculate pupil centers
                        left_moments = cv2.moments(left_pupil)
                        right_moments = cv2.moments(right_pupil)
                        
                        # Skip if moments are invalid
                        if left_moments["m00"] != 0 and right_moments["m00"] != 0:
                            left_pupil_x = left_moments["m10"] / left_moments["m00"] + left_eye_center[0] - 15
                            left_pupil_y = left_moments["m01"] / left_moments["m00"] + left_eye_center[1] - 10
                            
                            right_pupil_x = right_moments["m10"] / right_moments["m00"] + right_eye_center[0] - 15
                            right_pupil_y = right_moments["m01"] / right_moments["m00"] + right_eye_center[1] - 10
                            
                            # Average position
                            pupil_x = (left_pupil_x + right_pupil_x) / 2
                            pupil_y = (left_pupil_y + right_pupil_y) / 2
                            
                            # Normalize to screen coordinates
                            if self.calibrated and hasattr(self, 'calibration_matrix'):
                                # Apply calibration transformation
                                point = np.array([pupil_x, pupil_y, 1.0])
                                transformed = np.dot(self.calibration_matrix, point)
                                screen_x, screen_y = transformed[0], transformed[1]
                                
                                # Calculate normalized coordinates
                                frame_width, frame_height = frame.shape[1], frame.shape[0]
                                normalized_x = screen_x / frame_width
                                normalized_y = screen_y / frame_height
                                
                                # Apply screen resolution with full range
                                screen_x = int(normalized_x * self.screen_resolution[0])
                                screen_y = int(normalized_y * self.screen_resolution[1])
                                
                                # Add random movement to better simulate real eye movement
                                # This helps testing by making the gaze point move more realistically
                                if self.mode == 'webcam':
                                    # Add slight randomness to make it more visible on screen
                                    screen_x += int(np.random.normal(0, 10))
                                    screen_y += int(np.random.normal(0, 10))
                                
                                # Clamp values to valid range
                                normalized_x = max(0.0, min(1.0, normalized_x))
                                normalized_y = max(0.0, min(1.0, normalized_y))
                                screen_x = max(0, min(self.screen_resolution[0], screen_x))
                                screen_y = max(0, min(self.screen_resolution[1], screen_y))
                            else:
                                # IMPROVED: Enhanced mapping for webcam mode
                                # Full screen mapping with better distribution
                                frame_width, frame_height = frame.shape[1], frame.shape[0]
                                
                                # Map eye center relative to face rather than absolute position
                                # This makes it follow your eye movements better
                                face_center_x = (face.left() + face.right()) / 2
                                face_center_y = (face.top() + face.bottom()) / 2
                                
                                # Calculate relative position from face center
                                rel_x = (pupil_x - face_center_x) / (face.width() / 2)  # -1 to 1 range
                                rel_y = (pupil_y - face_center_y) / (face.height() / 2)  # -1 to 1 range
                                
                                # Scale to screen coordinates with enhanced sensitivity
                                # Map to full screen dimensions with increased sensitivity for small movements
                                sensitivity = 2.5  # Increase for more movement 
                                screen_x = int(self.screen_resolution[0] * (0.5 + (rel_x * sensitivity / 2)))
                                screen_y = int(self.screen_resolution[1] * (0.5 + (rel_y * sensitivity / 2)))
                                
                                # Normalized coordinates (0-1 range)
                                normalized_x = screen_x / self.screen_resolution[0]
                                normalized_y = screen_y / self.screen_resolution[1]
                                
                                # Add some randomness to simulate eye movement variance
                                if self.mode == 'webcam':
                                    screen_x += int(np.random.normal(0, 5))  # Reduced from 10 to 5
                                    screen_y += int(np.random.normal(0, 5))
                                
                                # Clamp values to screen bounds
                                normalized_x = max(0.0, min(1.0, normalized_x))
                                normalized_y = max(0.0, min(1.0, normalized_y))
                                screen_x = max(0, min(self.screen_resolution[0], screen_x))
                                screen_y = max(0, min(self.screen_resolution[1], screen_y))
                            
                            # Calculate pupil size (diameter) - simplified
                            left_pupil_area = cv2.contourArea(left_pupil)
                            right_pupil_area = cv2.contourArea(right_pupil)
                            pupil_size = (np.sqrt(left_pupil_area / np.pi) + np.sqrt(right_pupil_area / np.pi)) / 2
                            
                            # Confidence based on how open the eyes are
                            confidence = min(1.0, ear * 2.5)  # Scale ear to 0-1 range
                            
                            # Create and store gaze event
                            with self.lock:
                                self.events.append(GazeEvent(
                                    timestamp=time.time(),
                                    event_type='fixation',  # Simplified - should detect saccades
                                    x=normalized_x,
                                    y=normalized_y,
                                    screen_x=int(screen_x),
                                    screen_y=int(screen_y),
                                    pupil_size=pupil_size,
                                    confidence=confidence
                                ))
            
            # Update current frame for calibration purposes
            self.current_frame = frame.copy()
            self.frame_ready.set()
            
            # Sleep to reduce CPU usage
            time.sleep(0.01)
    
    def _generate_dummy_data(self):
        """
        Generate synthetic gaze data for the dummy tracker
        
        This creates realistic eye movement patterns including fixations,
        saccades and blinks without requiring actual eye tracking hardware
        """
        # Parameters for the random walk
        step_size = 0.01
        fixation_duration = 0.5  # seconds
        saccade_probability = 0.1
        blink_probability = 0.01
        
        # Current position
        x, y = 0.5, 0.5  # Start at center
        
        last_fixation_time = time.time()
        fixation_id = 0
        
        while self.running:
            current_time = time.time()
            
            # Random blink
            if np.random.random() < blink_probability:
                with self.lock:
                    self.events.append(GazeEvent(
                        timestamp=current_time,
                        event_type='blink',
                        confidence=1.0
                    ))
                    self.blink_count += 1
                    
                # Pause during blink
                time.sleep(0.1)
                continue
            
            # Check if it's time for a new fixation or continue current one
            elapsed = current_time - last_fixation_time
            
            if elapsed >= fixation_duration or np.random.random() < saccade_probability:
                # End current fixation
                if elapsed > 0.1:  # Only count as fixation if it lasted a bit
                    with self.lock:
                        self.events.append(GazeEvent(
                            timestamp=current_time,
                            event_type='fixation',
                            x=x,
                            y=y,
                            screen_x=int(x * self.screen_resolution[0]),
                            screen_y=int(y * self.screen_resolution[1]),
                            duration=elapsed,
                            pupil_size=3.5 + 0.5 * np.random.random(),  # Random pupil size between 3.5-4.0mm
                            confidence=0.8 + 0.2 * np.random.random()  # High confidence
                        ))
                        self.fixation_count += 1
                        self.total_fixation_duration += elapsed
                
                # Make a saccade
                with self.lock:
                    self.events.append(GazeEvent(
                        timestamp=current_time,
                        event_type='saccade',
                        confidence=0.7 + 0.3 * np.random.random()
                    ))
                    self.saccade_count += 1
                
                # Random new position (more realistic than small steps)
                x = np.random.random()
                y = np.random.random()
                
                # Start new fixation
                last_fixation_time = current_time
                fixation_id += 1
            else:
                # Small random walk during fixation (microsaccades)
                x += step_size * np.random.normal()
                y += step_size * np.random.normal()
                
                # Ensure x and y stay within bounds
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                
                # Add some samples during fixation
                with self.lock:
                    self.events.append(GazeEvent(
                        timestamp=current_time,
                        event_type='fixation',
                        x=x,
                        y=y,
                        screen_x=int(x * self.screen_resolution[0]),
                        screen_y=int(y * self.screen_resolution[1]),
                        pupil_size=3.5 + 0.5 * np.random.random(),  # Random pupil size
                        confidence=0.8 + 0.2 * np.random.random()  # High confidence
                    ))
            
            # Sleep to control sample rate
            time.sleep(1/60)  # ~60Hz sampling rate
    
    def calibrate(self, calibration_points: List[Tuple[float, float]] = None):
        """
        Calibrate the gaze tracker
        
        This aligns the raw eye tracking data with actual screen positions
        to ensure accurate gaze point detection
        
        Args:
            calibration_points: List of screen positions to use for calibration (normalized 0-1)
                                If None, default calibration points will be used
        
        Returns:
            bool: True if calibration successful, False otherwise
        """
        if self.mode == 'dummy':
            # Dummy tracker doesn't need calibration
            self.calibrated = True
            logger.info("Dummy tracker calibration skipped")
            return True
            
        if self.mode == 'tobii':
            try:
                import tobii_research as tr
                
                if calibration_points is None:
                    # Default 5-point calibration
                    calibration_points = [
                        (0.5, 0.5),  # Center
                        (0.1, 0.1),  # Top-left
                        (0.9, 0.1),  # Top-right
                        (0.9, 0.9),  # Bottom-right
                        (0.1, 0.9)   # Bottom-left
                    ]
                
                # Create calibration object
                calibration = tr.ScreenBasedCalibration(self.tracker)
                
                # Enter calibration mode
                calibration.enter_calibration_mode()
                
                # Collect data for each point
                for i, point in enumerate(calibration_points):
                    logger.info(f"Calibrating point {i+1}/{len(calibration_points)}: {point}")
                    
                    # Convert normalized point to screen coordinates
                    x, y = point
                    screen_x = x * self.screen_resolution[0]
                    screen_y = y * self.screen_resolution[1]
                    
                    # Collect data for this point 
                    # (in practice, you would show a visual stimulus here)
                    time.sleep(1.0)  # Give user time to fixate
                    
                    calibration_point = tr.Point2D(x, y)
                    status = calibration.collect_data(calibration_point)
                    
                    if status != tr.CALIBRATION_STATUS_SUCCESS:
                        logger.warning(f"Calibration failed for point {i+1}")
                
                # Compute and apply calibration
                calibration_result = calibration.compute_and_apply()
                
                # Check calibration results
                if calibration_result.status == tr.CALIBRATION_STATUS_SUCCESS:
                    logger.info("Calibration successful")
                    self.calibrated = True
                else:
                    logger.warning("Calibration failed")
                    self.calibrated = False
                
                # Exit calibration mode
                calibration.leave_calibration_mode()
                
                return self.calibrated
                
            except Exception as e:
                logger.error(f"Error during Tobii calibration: {e}")
                return False
                
        elif self.mode == 'webcam':
            # Webcam calibration is more involved and would require:
            # 1. Showing points on screen
            # 2. Detecting pupils while user looks at each point
            # 3. Building a mapping between pupil position and screen coordinates
            
            if not hasattr(self, 'current_frame') or self.current_frame is None:
                logger.error("No frames available for calibration")
                return False
            
            if calibration_points is None:
                # Default 9-point calibration
                calibration_points = [
                    (0.5, 0.5),  # Center
                    (0.1, 0.1),  # Top-left
                    (0.5, 0.1),  # Top-center
                    (0.9, 0.1),  # Top-right
                    (0.1, 0.5),  # Mid-left
                    (0.9, 0.5),  # Mid-right
                    (0.1, 0.9),  # Bottom-left
                    (0.5, 0.9),  # Bottom-center
                    (0.9, 0.9)   # Bottom-right
                ]
            
            # Store calibration points
            self.calibration_points = calibration_points
            
            # In a real application, you would implement:
            # - A UI to show calibration points
            # - Collection of eye positions at each point
            # - Calculation of a mapping (e.g., homography)
            
            # For now, we'll just simulate a successful calibration
            # with a diagonal calibration matrix
            self.calibration_matrix = np.array([
                [self.screen_resolution[0], 0, 0],
                [0, self.screen_resolution[1], 0],
                [0, 0, 1]
            ])
            
            self.calibrated = True
            logger.info("Simulated webcam calibration completed")
            return True
    
    def start(self) -> bool:
        """
        Begin tracking gaze events
        
        This initializes the appropriate tracking method and begins 
        collecting eye movement data
        
        Returns:
            bool: True if tracking started successfully, False otherwise
        """
        if self.running:
            logger.warning("Gaze tracking already started")
            return True
        
        if not self.calibrated and self.mode != 'dummy':
            logger.warning("Tracker not calibrated. Running calibration...")
            self.calibrate()
        
        self.running = True
        self.start_time = time.time()
        logger.info("Starting gaze tracking")
        
        if self.mode == 'webcam':
            # Start the frame processing thread
            self.stop_requested = False
            self.processing_thread = threading.Thread(target=self._process_webcam_frames)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
        elif self.mode == 'tobii':
            try:
                import tobii_research as tr
                # Subscribe to gaze data stream
                self.tracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, self.gaze_callback)
                logger.info("Subscribed to Tobii gaze data")
            except Exception as e:
                logger.error(f"Error starting Tobii tracker: {e}")
                self.running = False
                return False
                
        elif self.mode == 'dummy':
            # Start the dummy data generation thread
            self.dummy_thread = threading.Thread(target=self._generate_dummy_data)
            self.dummy_thread.daemon = True
            self.dummy_thread.start()
        
        return True
    
    def stop(self) -> None:
        """
        Stop tracking gaze events and save all collected data
        """
        if not self.running:
            logger.warning("Gaze tracking not started")
            return
        
        self.running = False
        logger.info("Stopping gaze tracking")
        
        if self.mode == 'webcam':
            # Stop the webcam processing thread
            self.stop_requested = True
            if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1.0)
            
            # Release the webcam
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
                
        elif self.mode == 'tobii':
            try:
                import tobii_research as tr
                # Unsubscribe from gaze data
                self.tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, self.gaze_callback)
                logger.info("Unsubscribed from Tobii gaze data")
            except Exception as e:
                logger.error(f"Error stopping Tobii tracker: {e}")
        
        # Save the data
        self.save_data()
    
    def save_data(self) -> None:
        """
        Save gaze tracking data to files
        
        This exports all collected eye tracking data to CSV and JSON formats
        for later analysis
        """
        if not self.events:
            logger.warning("No gaze events to save")
            return
        
        # Create timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, f"gaze_events_{timestamp}.csv")
        
        with self.lock:
            events_to_save = self.events.copy()
            self.events = []
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'event_type', 'x', 'y', 'screen_x', 'screen_y', 
                         'duration', 'pupil_size', 'confidence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for event in events_to_save:
                writer.writerow(asdict(event))
        
        # Also save to JSON for easier analysis
        json_path = os.path.join(self.output_dir, f"gaze_events_{timestamp}.json")
        with open(json_path, 'w') as jsonfile:
            json.dump([asdict(event) for event in events_to_save], jsonfile, indent=2)
        
        duration = time.time() - self.start_time
        logger.info(f"Saved {len(events_to_save)} gaze events to {csv_path} and {json_path}")
        logger.info(f"Total tracking duration: {duration:.2f} seconds")
        logger.info(f"Total fixations: {self.fixation_count}")
        logger.info(f"Total saccades: {self.saccade_count}")
        logger.info(f"Total blinks: {self.blink_count}")

def main():
    """
    Entry point when running this script directly
    Parses command line arguments and manages the tracking process so
    it can be run from the command line with different options
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Gaze Tracker')
    parser.add_argument('--output', type=str, default='mouse_data',
                        help='Output directory for gaze data')
    parser.add_argument('--mode', type=str, default=GAZE_TRACKER_MODE,
                        choices=['webcam', 'tobii', 'dummy'],
                        help='Tracking mode (webcam, tobii, or dummy)')
    parser.add_argument('--duration', type=int, default=0,
                        help='Duration to track in seconds (0 for indefinite)')
    args = parser.parse_args()
    
    tracker = GazeTracker(output_dir=args.output, mode=args.mode)
    
    try:
        # Calibrate if needed
        if args.mode != 'dummy':
            tracker.calibrate()
        
        # Start tracking
        tracker.start()
        logger.info(f"Gaze tracking started. Press Ctrl+C to stop.")
        
        if args.duration > 0:
            # Run for the specified duration
            time.sleep(args.duration)
            tracker.stop()
        else:
            # Run indefinitely until manually stopped so the user can stop it with Ctrl+C
            while tracker.running:
                time.sleep(1)
    
    except KeyboardInterrupt:
        # Handle when user presses Ctrl+C
        logger.info("Keyboard interrupt received")
    finally:
        # Always make sure to stop tracking and save data
        tracker.stop()
        logger.info("Gaze tracking completed")

# This section will run when the script is executed directly (not imported)
if __name__ == "__main__":
    main() 