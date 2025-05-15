#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ======================================================
# Sync Tracker - Combined mouse and gaze data engine
# 
# This core synchronization engine:
# - Integrates mouse and eye tracking data streams
# - Calculates metrics like eye-hand coordination
# - Generates visualizations comparing both inputs
# - Provides temporal analysis of attention patterns
# - Creates comprehensive HTML and CSV reports
# 
# The foundation for advanced HCI research that
# studies the relationship between looking and pointing
# ======================================================

import time
import datetime
import os
import threading
import logging
import numpy as np
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from matplotlib.figure import Figure

# Import our tracking modules
from mouse_tracker import MouseTracker, MouseEvent
from gaze_tracker import GazeTracker, GazeEvent

# Get environment variables
DEFAULT_DATA_DIR = os.environ.get('MOUSE_TRACKER_DATA_DIR', os.path.join(os.getcwd(), "mouse_data"))
DEBUG_MODE = os.environ.get('MOUSE_TRACKER_DEBUG', 'false').lower() == 'true'
GAZE_TRACKER_MODE = os.environ.get('GAZE_TRACKER_MODE', 'dummy').lower()  # 'webcam', 'tobii', or 'dummy'

# Set up logging
log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(os.path.join("logs", "sync_tracker.log")), logging.StreamHandler()]
)
logger = logging.getLogger("SyncTracker")

@dataclass
class SyncEvent:
    """Data class for synchronized mouse and gaze events"""
    # This structure combines data from both tracking systems
    # into a unified format for analysis and visualization
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

class SyncTracker:
    """Class that synchronizes mouse and gaze tracking"""
    
    def __init__(self, output_dir: str = DEFAULT_DATA_DIR,
                 gaze_mode: str = GAZE_TRACKER_MODE,
                 screen_resolution: Tuple[int, int] = None):
        """Initialize the synchronized tracker
        
        Args:
            output_dir: Directory to save tracking data
            gaze_mode: Gaze tracking mode ('webcam', 'tobii', or 'dummy')
            screen_resolution: Monitor resolution (width, height)
        """
        # This creates a unified tracking system that combines
        # mouse and gaze data streams with synchronized timestamps
        self.output_dir = output_dir
        self.gaze_mode = gaze_mode
        
        # Get screen resolution
        # This auto-detects the display size for proper coordinate mapping
        if screen_resolution is None:
            try:
                import tkinter as tk
                root = tk.Tk()
                screen_width = root.winfo_screenwidth()
                screen_height = root.winfo_screenheight()
                root.destroy()
                self.screen_resolution = (screen_width, screen_height)
            except:
                # Default resolution if we can't detect
                self.screen_resolution = (1920, 1080)
                logger.warning("Could not detect screen resolution. Using default: 1920x1080")
        else:
            self.screen_resolution = screen_resolution
        
        # Initialize trackers
        # Both mouse and gaze trackers run independently but are synchronized
        self.mouse_tracker = MouseTracker(output_dir=output_dir)
        self.gaze_tracker = GazeTracker(output_dir=output_dir, 
                                        mode=gaze_mode,
                                        screen_resolution=self.screen_resolution)
        
        # Synchronization data
        # This stores the combined stream of both tracking systems
        self.sync_events: List[SyncEvent] = []
        self.lock = threading.Lock()
        self.running = False
        self.sync_thread = None
        self.start_time = 0
        
        # Analysis metrics
        # These track the relationship between gaze and mouse position
        self.total_distance = 0.0
        self.avg_distance = 0.0
        self.max_distance = 0.0
        self.attention_match_percent = 0.0  # % of time gaze is near mouse
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"SyncTracker initialized with gaze mode: {gaze_mode}")
        logger.debug(f"Using output directory: {self.output_dir}")
        logger.debug(f"Screen resolution: {self.screen_resolution}")
    
    def _sync_events_thread(self):
        """Thread function to synchronize mouse and gaze events"""
        # This background thread continuously integrates data from both tracking systems
        # It runs at a fixed rate to ensure consistent timing and CPU usage
        last_processed_mouse_idx = 0
        last_processed_gaze_idx = 0
        sync_interval = 1/30  # 30Hz synchronization rate
        
        while self.running:
            current_time = time.time()
            
            # Get recent mouse events
            # This collects any new mouse data since the last check
            with self.mouse_tracker.lock:
                mouse_events = self.mouse_tracker.events[last_processed_mouse_idx:]
                last_processed_mouse_idx = len(self.mouse_tracker.events)
            
            # Get recent gaze events
            # This collects any new gaze data since the last check
            with self.gaze_tracker.lock:
                gaze_events = self.gaze_tracker.events[last_processed_gaze_idx:]
                last_processed_gaze_idx = len(self.gaze_tracker.events)
            
            # Process mouse events
            # Each mouse event is converted to a synchronized event format
            for event in mouse_events:
                # Convert MouseEvent to SyncEvent
                if event.event_type == 'move':
                    sync_event = SyncEvent(
                        timestamp=event.timestamp,
                        event_type='mouse_move',
                        mouse_x=event.x,
                        mouse_y=event.y
                    )
                    self._add_sync_event(sync_event)
                elif event.event_type == 'click':
                    sync_event = SyncEvent(
                        timestamp=event.timestamp,
                        event_type='mouse_click',
                        mouse_x=event.x,
                        mouse_y=event.y,
                        mouse_button=event.button,
                        mouse_pressed=event.pressed
                    )
                    self._add_sync_event(sync_event)
                elif event.event_type == 'scroll':
                    sync_event = SyncEvent(
                        timestamp=event.timestamp,
                        event_type='mouse_scroll',
                        mouse_x=event.x,
                        mouse_y=event.y,
                        mouse_dx=event.dx,
                        mouse_dy=event.dy
                    )
                    self._add_sync_event(sync_event)
            
            # Process gaze events
            # Each gaze event is converted to a synchronized event format
            for event in gaze_events:
                # Convert GazeEvent to SyncEvent
                if event.event_type == 'fixation':
                    sync_event = SyncEvent(
                        timestamp=event.timestamp,
                        event_type='gaze_fixation',
                        gaze_x=event.x,
                        gaze_y=event.y,
                        gaze_screen_x=event.screen_x,
                        gaze_screen_y=event.screen_y,
                        gaze_duration=event.duration,
                        gaze_pupil_size=event.pupil_size,
                        gaze_confidence=event.confidence
                    )
                    self._add_sync_event(sync_event)
                elif event.event_type == 'saccade':
                    sync_event = SyncEvent(
                        timestamp=event.timestamp,
                        event_type='gaze_saccade',
                        gaze_confidence=event.confidence
                    )
                    self._add_sync_event(sync_event)
                elif event.event_type == 'blink':
                    sync_event = SyncEvent(
                        timestamp=event.timestamp,
                        event_type='gaze_blink',
                        gaze_confidence=event.confidence
                    )
                    self._add_sync_event(sync_event)
            
            # Sleep to control sync rate
            time.sleep(sync_interval)
    
    def _add_sync_event(self, event: SyncEvent):
        """Add an event to the synchronized event list with distance calculation"""
        # This enriches the synchronized events with distance metrics
        # between mouse and gaze positions when both are available
        # Find the most recent mouse and gaze positions
        recent_mouse_pos = None
        recent_gaze_pos = None
        
        # Look for recent positions in existing events
        with self.lock:
            for e in reversed(self.sync_events[-20:] if len(self.sync_events) > 20 else self.sync_events):
                # Find most recent mouse position
                if recent_mouse_pos is None and e.mouse_x is not None and e.mouse_y is not None:
                    recent_mouse_pos = (e.mouse_x, e.mouse_y)
                
                # Find most recent gaze position
                if recent_gaze_pos is None and e.gaze_screen_x is not None and e.gaze_screen_y is not None:
                    recent_gaze_pos = (e.gaze_screen_x, e.gaze_screen_y)
                
                # If we found both, we can stop looking
                if recent_mouse_pos is not None and recent_gaze_pos is not None:
                    break
        
        # Update the event with the recent positions if they're missing
        if event.event_type.startswith('mouse_'):
            # This is a mouse event, update gaze position
            if event.gaze_screen_x is None and event.gaze_screen_y is None and recent_gaze_pos is not None:
                event.gaze_screen_x = recent_gaze_pos[0]
                event.gaze_screen_y = recent_gaze_pos[1]
                
                # Also calculate normalized gaze coordinates
                if self.screen_resolution[0] > 0 and self.screen_resolution[1] > 0:
                    event.gaze_x = event.gaze_screen_x / self.screen_resolution[0]
                    event.gaze_y = event.gaze_screen_y / self.screen_resolution[1]
        
        elif event.event_type.startswith('gaze_'):
            # This is a gaze event, update mouse position
            if event.mouse_x is None and event.mouse_y is None and recent_mouse_pos is not None:
                event.mouse_x = recent_mouse_pos[0]
                event.mouse_y = recent_mouse_pos[1]
        
        # Calculate distance between mouse and gaze if we have both positions
        if (event.mouse_x is not None and event.mouse_y is not None and 
            event.gaze_screen_x is not None and event.gaze_screen_y is not None):
            
            event.distance = np.sqrt(
                (event.mouse_x - event.gaze_screen_x) ** 2 +
                (event.mouse_y - event.gaze_screen_y) ** 2
            )
            
            # Calculate normalized distance (relative to screen diagonal)
            screen_diagonal = np.sqrt(self.screen_resolution[0] ** 2 + self.screen_resolution[1] ** 2)
            event.normalized_distance = event.distance / screen_diagonal if screen_diagonal > 0 else 0
            
            # Update metrics
            self.total_distance += event.distance
            self.max_distance = max(self.max_distance, event.distance)
            
            # Update count of samples for average calculation
            if len(self.sync_events) > 0:
                self.avg_distance = self.total_distance / (len(self.sync_events) + 1)
        
        # Add the event to our list
        with self.lock:
            self.sync_events.append(event)
    
    def start(self) -> bool:
        """Start tracking mouse and gaze movements synchronously"""
        # This activates both tracking systems and starts the synchronization thread
        # Returns success/failure status to indicate if tracking started properly
        if self.running:
            logger.warning("Synchronized tracking already started")
            return True
        
        logger.info("Starting synchronized tracking")
        
        # Start mouse tracker
        mouse_started = self.mouse_tracker.start()
        if not mouse_started:
            logger.error("Failed to start mouse tracker")
            return False
        
        # Start gaze tracker
        gaze_started = self.gaze_tracker.start()
        if not gaze_started:
            logger.warning("Failed to start gaze tracker. Only mouse tracking will be available.")
        
        # Start synchronization thread
        self.running = True
        self.start_time = time.time()
        self.sync_thread = threading.Thread(target=self._sync_events_thread)
        self.sync_thread.daemon = True
        self.sync_thread.start()
        
        return True
    
    def stop(self) -> None:
        """Stop tracking and save data"""
        # This gracefully shuts down all tracking systems and saves collected data
        # It calculates final metrics before stopping the synchronization process
        if not self.running:
            logger.warning("Synchronized tracking not started")
            return
        
        logger.info("Stopping synchronized tracking")
        
        # Stop synchronization thread
        self.running = False
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=1.0)
        
        # Stop trackers
        self.mouse_tracker.stop()
        self.gaze_tracker.stop()
        
        # Save the synchronized data
        self.save_data()
        
        # Calculate final metrics
        self._calculate_metrics()
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate metrics about the relationship between mouse and gaze"""
        # This analyzes the relationship between eye and mouse movements
        # It generates metrics about coordination, attention, and behavior patterns
        # Already calculated during tracking:
        # - total_distance
        # - avg_distance
        # - max_distance
        
        # Calculate attention match percentage
        # (percent of time gaze is within 100 pixels of mouse)
        attention_threshold = 100  # pixels
        attention_matches = sum(1 for e in self.sync_events if e.distance is not None and e.distance < attention_threshold)
        total_events_with_distance = sum(1 for e in self.sync_events if e.distance is not None)
        
        if total_events_with_distance > 0:
            self.attention_match_percent = (attention_matches / total_events_with_distance) * 100
        
        # Log metrics
        logger.info(f"Average mouse-gaze distance: {self.avg_distance:.2f} pixels")
        logger.info(f"Maximum mouse-gaze distance: {self.max_distance:.2f} pixels")
        logger.info(f"Attention match percentage: {self.attention_match_percent:.2f}%")
        
        return {
            "avg_distance": self.avg_distance,
            "max_distance": self.max_distance,
            "attention_match_percent": self.attention_match_percent
        }
    
    def save_data(self) -> Tuple[str, str]:
        """Save synchronized data to JSON and CSV files"""
        # This exports the collected data in both structured formats
        # JSON preserves full detail, while CSV enables easy spreadsheet analysis
        if not self.sync_events:
            logger.warning("No synchronized events to save")
            return ("", "")
        
        # Create timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, f"sync_events_{timestamp}.csv")
        
        with self.lock:
            events_to_save = self.sync_events.copy()
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                'timestamp', 'event_type', 
                'mouse_x', 'mouse_y', 
                'gaze_x', 'gaze_y', 
                'gaze_screen_x', 'gaze_screen_y',
                'mouse_button', 'mouse_pressed', 
                'mouse_dx', 'mouse_dy',
                'gaze_duration', 'gaze_pupil_size', 'gaze_confidence',
                'distance', 'normalized_distance'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for event in events_to_save:
                writer.writerow(asdict(event))
        
        # Also save to JSON for easier analysis
        json_path = os.path.join(self.output_dir, f"sync_events_{timestamp}.json")
        with open(json_path, 'w') as jsonfile:
            json.dump([asdict(event) for event in events_to_save], jsonfile, indent=2)
        
        duration = time.time() - self.start_time
        logger.info(f"Saved {len(events_to_save)} synchronized events to {csv_path} and {json_path}")
        logger.info(f"Total tracking duration: {duration:.2f} seconds")
        
        return (csv_path, json_path)
    
    def generate_heatmap_comparison(self) -> str:
        """Generate a heatmap comparing mouse and gaze positions"""
        # This creates a side-by-side heatmap visualization showing
        # where the mouse moved vs. where the user looked
        if not self.sync_events:
            logger.warning("No data to generate heatmap comparison")
            return ""
        
        # Extract mouse and gaze positions
        mouse_positions = [(e.mouse_x, e.mouse_y) for e in self.sync_events 
                           if e.mouse_x is not None and e.mouse_y is not None]
        
        gaze_positions = [(e.gaze_screen_x, e.gaze_screen_y) for e in self.sync_events 
                          if e.gaze_screen_x is not None and e.gaze_screen_y is not None]
        
        if not mouse_positions or not gaze_positions:
            logger.warning("Insufficient data to generate heatmap comparison")
            return ""
        
        # Convert to numpy arrays
        mouse_x, mouse_y = zip(*mouse_positions)
        gaze_x, gaze_y = zip(*gaze_positions)
        
        # Create figure with two subplots
        plt.figure(figsize=(15, 7))
        
        # Mouse heatmap
        plt.subplot(1, 2, 1)
        plt.hist2d(mouse_x, mouse_y, bins=50, cmap='hot')
        plt.colorbar(label='Frequency')
        plt.title('Mouse Position Heatmap')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.gca().invert_yaxis()  # Invert y-axis to match screen coordinates
        
        # Gaze heatmap
        plt.subplot(1, 2, 2)
        plt.hist2d(gaze_x, gaze_y, bins=50, cmap='hot')
        plt.colorbar(label='Frequency')
        plt.title('Gaze Position Heatmap')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.gca().invert_yaxis()  # Invert y-axis to match screen coordinates
        
        # Save the figure
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = os.path.join(self.output_dir, f"heatmap_comparison_{timestamp}.png")
        plt.savefig(fig_path)
        plt.close()
        
        logger.info(f"Heatmap comparison saved to {fig_path}")
        return fig_path
    
    def generate_distance_plot(self) -> str:
        """Generate a plot showing distance between mouse and gaze over time"""
        # This visualizes the coordination between eye and hand
        # Lower distances indicate tighter eye-hand coordination
        if not self.sync_events:
            logger.warning("No data to generate distance plot")
            return ""
        
        # Extract timestamps and distances
        timestamps = []
        distances = []
        
        for event in self.sync_events:
            if event.distance is not None:
                timestamps.append(event.timestamp)
                distances.append(event.distance)
        
        if not timestamps or not distances:
            logger.warning("No distance data to plot")
            return ""
        
        # Convert timestamps to relative time in seconds
        start_time = timestamps[0]
        relative_times = [(t - start_time) for t in timestamps]
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(relative_times, distances, '-', alpha=0.5, linewidth=1)
        
        # Add a smoothed trend line
        if len(relative_times) > 10:
            try:
                import scipy.signal
                smoothed = scipy.signal.savgol_filter(distances, min(51, len(distances) // 2 * 2 + 1), 3)
                plt.plot(relative_times, smoothed, 'r-', linewidth=2, label='Smoothed')
                plt.legend()
            except ImportError:
                # If scipy is not available, use a simple moving average
                window_size = min(20, len(distances) // 5)
                if window_size > 1:
                    smoothed = np.convolve(distances, np.ones(window_size)/window_size, mode='valid')
                    valid_times = relative_times[window_size-1:]
                    if len(valid_times) == len(smoothed):
                        plt.plot(valid_times, smoothed, 'r-', linewidth=2, label='Smoothed')
                        plt.legend()
        
        plt.title('Mouse-Gaze Distance Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Distance (pixels)')
        plt.grid(True, alpha=0.3)
        
        # Add horizontal line for attention threshold
        plt.axhline(y=100, color='g', linestyle='--', alpha=0.7, label='Attention Threshold (100px)')
        plt.legend()
        
        # Save the figure
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = os.path.join(self.output_dir, f"distance_plot_{timestamp}.png")
        plt.savefig(fig_path)
        plt.close()
        
        logger.info(f"Distance plot saved to {fig_path}")
        return fig_path
    
    def generate_trajectory_comparison(self) -> str:
        """Generate a trajectory plot comparing mouse and gaze paths"""
        # This shows the actual paths taken by both tracking systems
        # allowing direct comparison of movement patterns
        if not self.sync_events:
            logger.warning("No data to generate trajectory comparison")
            return ""
        
        # Extract mouse and gaze positions
        mouse_positions = [(e.mouse_x, e.mouse_y) for e in self.sync_events 
                           if e.mouse_x is not None and e.mouse_y is not None]
        
        gaze_positions = [(e.gaze_screen_x, e.gaze_screen_y) for e in self.sync_events 
                          if e.gaze_screen_x is not None and e.gaze_screen_y is not None]
        
        if not mouse_positions or not gaze_positions:
            logger.warning("Insufficient data to generate trajectory comparison")
            return ""
        
        # Sample the data if there are too many points
        max_points = 1000
        if len(mouse_positions) > max_points:
            step = len(mouse_positions) // max_points
            mouse_positions = mouse_positions[::step]
        
        if len(gaze_positions) > max_points:
            step = len(gaze_positions) // max_points
            gaze_positions = gaze_positions[::step]
        
        # Convert to numpy arrays
        mouse_x, mouse_y = zip(*mouse_positions)
        gaze_x, gaze_y = zip(*gaze_positions)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        plt.plot(mouse_x, mouse_y, 'b-', alpha=0.5, linewidth=1, label='Mouse')
        plt.plot(gaze_x, gaze_y, 'r-', alpha=0.5, linewidth=1, label='Gaze')
        
        # Add arrows to show direction
        arrow_interval = max(len(mouse_positions) // 10, 1)
        for i in range(0, len(mouse_positions) - 1, arrow_interval):
            plt.arrow(mouse_x[i], mouse_y[i], 
                     mouse_x[i+1] - mouse_x[i], mouse_y[i+1] - mouse_y[i],
                     head_width=15, head_length=15, fc='blue', ec='blue', alpha=0.3)
        
        arrow_interval = max(len(gaze_positions) // 10, 1)
        for i in range(0, len(gaze_positions) - 1, arrow_interval):
            plt.arrow(gaze_x[i], gaze_y[i], 
                     gaze_x[i+1] - gaze_x[i], gaze_y[i+1] - gaze_y[i],
                     head_width=15, head_length=15, fc='red', ec='red', alpha=0.3)
        
        plt.title('Mouse and Gaze Trajectories')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Invert y-axis to match screen coordinates
        plt.gca().invert_yaxis()
        
        # Set limits based on screen resolution
        plt.xlim(0, self.screen_resolution[0])
        plt.ylim(self.screen_resolution[1], 0)
        
        # Save the figure
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = os.path.join(self.output_dir, f"trajectory_comparison_{timestamp}.png")
        plt.savefig(fig_path)
        plt.close()
        
        logger.info(f"Trajectory comparison saved to {fig_path}")
        return fig_path
    
    def generate_report(self) -> str:
        """Generate a comprehensive HTML report of synchronized tracking"""
        # This creates a complete analysis with all visualizations and metrics
        # The HTML format makes it easy to view and share findings
        if not self.sync_events:
            logger.warning("No data to generate report")
            return ""
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Generate visualizations
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        heatmap_path = self.generate_heatmap_comparison()
        distance_path = self.generate_distance_plot()
        trajectory_path = self.generate_trajectory_comparison()
        
        # Determine total duration
        if len(self.sync_events) >= 2:
            start_time = self.sync_events[0].timestamp
            end_time = self.sync_events[-1].timestamp
            duration_seconds = end_time - start_time
            duration_formatted = str(datetime.timedelta(seconds=int(duration_seconds)))
        else:
            duration_seconds = 0
            duration_formatted = "0:00:00"
        
        # Count events by type
        event_counts = {}
        for event in self.sync_events:
            if event.event_type in event_counts:
                event_counts[event.event_type] += 1
            else:
                event_counts[event.event_type] = 1
        
        # Create HTML report
        report_path = os.path.join(self.output_dir, f"sync_report_{timestamp}.html")
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Mouse and Gaze Tracking Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .metric {{ margin-bottom: 5px; }}
                .metric-name {{ font-weight: bold; }}
                .metric-value {{ margin-left: 10px; }}
                .section {{ margin-bottom: 30px; }}
                img {{ max-width: 100%; border: 1px solid #ddd; margin-top: 10px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>Mouse and Gaze Tracking Report</h1>
            <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Summary</h2>
                <div class="metric">
                    <span class="metric-name">Duration:</span>
                    <span class="metric-value">{duration_formatted}</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Total Events:</span>
                    <span class="metric-value">{len(self.sync_events)}</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Average Mouse-Gaze Distance:</span>
                    <span class="metric-value">{metrics['avg_distance']:.2f} pixels</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Maximum Mouse-Gaze Distance:</span>
                    <span class="metric-value">{metrics['max_distance']:.2f} pixels</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Attention Match Percentage:</span>
                    <span class="metric-value">{metrics['attention_match_percent']:.2f}%</span>
                </div>
            </div>
            
            <div class="section">
                <h2>Event Distribution</h2>
                <table>
                    <tr>
                        <th>Event Type</th>
                        <th>Count</th>
                    </tr>
        """
        
        # Add event counts to table
        for event_type, count in event_counts.items():
            html_content += f"""
                    <tr>
                        <td>{event_type}</td>
                        <td>{count}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
        """
        
        # Add visualizations if available
        if heatmap_path:
            html_content += f"""
                <h3>Mouse and Gaze Heatmap Comparison</h3>
                <img src="{os.path.basename(heatmap_path)}" alt="Heatmap Comparison">
            """
        
        if distance_path:
            html_content += f"""
                <h3>Mouse-Gaze Distance Over Time</h3>
                <img src="{os.path.basename(distance_path)}" alt="Distance Plot">
            """
        
        if trajectory_path:
            html_content += f"""
                <h3>Mouse and Gaze Trajectories</h3>
                <img src="{os.path.basename(trajectory_path)}" alt="Trajectory Comparison">
            """
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>Interpretation</h2>
                <p>The attention match percentage indicates how often the user's gaze is near the mouse cursor. A higher percentage suggests the user is actively following the mouse with their eyes.</p>
                <p>The mouse-gaze distance metrics show how far apart the user's gaze and cursor typically are. Lower values suggest tighter coordination between eye movements and mouse control.</p>
                <p>The heatmap comparison reveals areas of the screen that receive the most mouse activity versus visual attention, which may not always overlap.</p>
            </div>
        </body>
        </html>
        """
        
        # Write the HTML file
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML report at {report_path}")
        return report_path
        
def main():
    """Run a basic synchronization demo"""
    # This provides a simple command-line demonstration
    # of the synchronization capabilities
    import argparse
    
    parser = argparse.ArgumentParser(description='Synchronized Mouse and Gaze Tracker')
    parser.add_argument('--output', type=str, default=DEFAULT_DATA_DIR,
                        help='Output directory for tracking data')
    parser.add_argument('--gaze-mode', type=str, default=GAZE_TRACKER_MODE,
                        choices=['webcam', 'tobii', 'dummy'],
                        help='Gaze tracking mode')
    parser.add_argument('--duration', type=int, default=0,
                        help='Duration to track in seconds (0 for indefinite)')
    parser.add_argument('--report', action='store_true',
                        help='Generate report after tracking')
    args = parser.parse_args()
    
    tracker = SyncTracker(output_dir=args.output, gaze_mode=args.gaze_mode)
    
    try:
        # Start tracking
        tracker.start()
        logger.info(f"Synchronized tracking started. Press Ctrl+C to stop.")
        
        if args.duration > 0:
            time.sleep(args.duration)
            tracker.stop()
        else:
            # Run indefinitely until interrupted
            while tracker.running:
                time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        tracker.stop()
        
        # Generate report if requested
        if args.report:
            report_path = tracker.generate_report()
            if report_path:
                logger.info(f"Report saved to {report_path}")
                
                # Try to open the report in a browser
                try:
                    import webbrowser
                    webbrowser.open(f"file://{os.path.abspath(report_path)}")
                except:
                    pass
        
        logger.info("Synchronized tracking completed")

if __name__ == "__main__":
    main() 