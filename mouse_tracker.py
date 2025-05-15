#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ======================================================
# Mouse Tracker - Advanced mouse movement tracking system
# 
# This program records all mouse activities including:
# - Mouse movements
# - Mouse clicks
# - Mouse scrolling
# 
# Data is automatically saved and can be visualized
# as heatmaps to show usage patterns
# ======================================================

import time 
import datetime 
import json 
import os 
import threading
import csv
import logging
import argparse 
from typing import Dict, List, Tuple, Any, Optional #
from pynput import mouse  # The key library that captures mouse events
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict

# ====== CONFIGURATION ======
# Default settings that can be overridden by environment variables
# This can be changed by setting the MOUSE_TRACKER_DATA_DIR environment variable
DEFAULT_DATA_DIR = os.environ.get('MOUSE_TRACKER_DATA_DIR', os.path.join(os.getcwd(), "mouse_data"))
DEBUG_MODE = os.environ.get('MOUSE_TRACKER_DEBUG', 'false').lower() == 'true'

# Setting up logging to track what's happening in the program
log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(os.path.join("logs", "mouse_tracker.log")), logging.StreamHandler()]
)
logger = logging.getLogger("MouseTracker")

@dataclass
class MouseEvent:
    """
    Data structure for storing mouse events
    Each event captures what the mouse did at a specific moment in time
    """
    timestamp: float         # When the event happened (seconds since epoch)
    event_type: str          # What kind of event: 'move', 'click', or 'scroll'
    x: int = 0               # X-coordinate on screen
    y: int = 0               # Y-coordinate on screen
    button: Optional[str] = None    # Which button was used (for clicks)
    pressed: Optional[bool] = None  # Was the button pressed or released?
    dx: Optional[int] = None        # Horizontal scroll amount
    dy: Optional[int] = None        # Vertical scroll amount

class MouseTracker:
    """
    The main tracking system that records and analyzes mouse behavior
    """
    
    def __init__(self, output_dir: str = DEFAULT_DATA_DIR):
        """
        Initialize the tracker with basic settings
        
        Args:
            output_dir: Where to save the collected data
        """
        # Storage for all mouse events
        self.events: List[MouseEvent] = []
        self.running = False
        self.output_dir = output_dir
        self.listener = None
        self.last_event_time = 0
        self.lock = threading.Lock()  # Prevents data corruption when multiple threads access data
        
        # Make sure we have a place to save our data
        os.makedirs(output_dir, exist_ok=True)
        
        # Tracking statistics
        self.total_distance = 0       # How far the mouse has moved (in pixels)
        self.last_position = (0, 0)   # Previous mouse position
        self.start_time = 0           # When tracking began
        
        logger.info("MouseTracker initialized")
        logger.debug(f"Using output directory: {self.output_dir}")
        logger.debug(f"Debug mode: {DEBUG_MODE}")
    
    def on_move(self, x: int, y: int) -> None:
        """
        Called whenever the mouse moves to a new position
        
        Args:
            x, y: The new mouse coordinates
        """
        timestamp = time.time()
        
        # Calculate how far the mouse has moved since last position
        if self.last_position != (0, 0): # Ignore the first move event
            distance = np.sqrt((x - self.last_position[0])**2 + (y - self.last_position[1])**2)
            self.total_distance += distance
        
        self.last_position = (x, y)
        
        # Store this movement event safely
        with self.lock: 
            self.events.append(MouseEvent(
                timestamp=timestamp,
                event_type='move',
                x=x,
                y=y
            ))
        
        # Don't flood the logs with movement messages means we only log every second 
        # Preventing too many log messages and reducing the amount of data we save
        if timestamp - self.last_event_time > 1.0:
            logger.debug(f"Mouse moved to {x}, {y}")
            self.last_event_time = timestamp
    
    def on_click(self, x: int, y: int, button: mouse.Button, pressed: bool) -> None:
        """
        Called whenever a mouse button is pressed or released
        
        Args:
            x, y: Mouse coordinates when the click happened
            button: Which button was clicked (left, right, middle)
            pressed: True for press, False for release
        """
        # Convert button names to a more readable format
        button_name = str(button).replace('Button.', '')
        timestamp = time.time()
        
        # Store this click event safely so we can access it from multiple threads
        with self.lock:
            self.events.append(MouseEvent(
                timestamp=timestamp,
                event_type='click',
                x=x,
                y=y,
                button=button_name,
                pressed=pressed
            ))
        
        # Clicks are important, so we always log them
        logger.info(f"Mouse {'pressed' if pressed else 'released'} {button_name} at {x}, {y}")
    
    def on_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        """
        Called whenever the mouse wheel is scrolled
        
        Args:
            x, y: Mouse coordinates during scrolling
            dx, dy: Amount scrolled horizontally and vertically
        """
        timestamp = time.time()
        
        # Store this scroll event safely
        with self.lock:
            self.events.append(MouseEvent(
                timestamp=timestamp,
                event_type='scroll',
                x=x,
                y=y,
                dx=dx,
                dy=dy
            ))
        
        logger.debug(f"Mouse scrolled at {x}, {y} by {dx}, {dy}")
    
    def start(self) -> bool:
        """
        Begin tracking mouse events
        
        Returns:
            True if tracking started successfully
        """
        # We prevent starting if already running
        if self.running:
            logger.warning("Tracking already started")
            return True
        
        self.running = True
        self.start_time = time.time()
        logger.info("Starting mouse tracking")
        
        # We Start listening for mouse events in the background
        self.listener = mouse.Listener(
            on_move=self.on_move,
            on_click=self.on_click,
            on_scroll=self.on_scroll
        )
        self.listener.start()
        
        # We start a background thread to save data periodically (prevents data loss) 
        # so when the program crashes or is closed, we still have some data saved
        self.save_thread = threading.Thread(target=self._periodic_save)
        self.save_thread.daemon = True  # This thread will exit when the program exits
        self.save_thread.start()
        
        return True
    
    def stop(self) -> None:
        """
        Stop tracking mouse events and save all collected data
        """
        if not self.running:
            logger.warning("Tracking not started")
            return
        
        self.running = False
        logger.info("Stopping mouse tracking")
        
        # Stop listening for mouse events
        if self.listener:
            self.listener.stop()
            self.listener.join()
        
        # Save all remaining data
        self.save_data()
    
    def _periodic_save(self) -> None:
        """
        Background task that saves data periodically to prevent data loss
        """
        while self.running:
            time.sleep(60)  # Wait for one minute
            if self.running:
                self.save_data(periodic=True)
    
    def save_data(self, periodic: bool = False) -> None:
        """
        Save all collected mouse events to files
        
        Args:
            periodic: If True, this is an automatic save; if False, it's a final save
        """
        if not self.events:
            logger.warning("No events to save")
            return
        
        # Create a timestamp for the filename (YYYYMMDD_HHMMSS format)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_suffix = "periodic" if periodic else "final"
        
        # Save to CSV (good for importing into spreadsheets)
        csv_path = os.path.join(self.output_dir, f"mouse_events_{timestamp}_{file_suffix}.csv")
        
        # Safely get a copy of the events to save
        with self.lock:
            events_to_save = self.events.copy()
            # If this is a final save, clear the events list
            if not periodic:
                self.events = []
        
        #Write events to CSV file for easy reading so this makes it easy to open in Excel or similar programs
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'event_type', 'x', 'y', 'button', 'pressed', 'dx', 'dy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for event in events_to_save:
                writer.writerow(asdict(event)) # Convert dataclass to dict for CSV writing
        
        # Also save to JSON format (better for programmatic analysis)
        json_path = os.path.join(self.output_dir, f"mouse_events_{timestamp}_{file_suffix}.json")
        with open(json_path, 'w') as jsonfile:
            json.dump([asdict(event) for event in events_to_save], jsonfile, indent=2)
        
        # Log some statistics about what we saved
        duration = time.time() - self.start_time
        logger.info(f"Saved {len(events_to_save)} events to {csv_path} and {json_path}")
        logger.info(f"Total tracking duration: {duration:.2f} seconds")
        logger.info(f"Total distance moved: {self.total_distance:.2f} pixels")

    def generate_heatmap(self, resolution: Tuple[int, int] = (1920, 1080), grid_size: int = 50) -> None:
        """
        Create a visual heatmap showing where the mouse spent most time
        
        Args:
            resolution: Screen resolution (width, height)
            grid_size: Detail level of the heatmap (higher = more detailed)
        """
        if not self.events:
            logger.warning("No events to generate heatmap")
            return
        
        # We only care about mouse movement for heatmaps
        move_events = [event for event in self.events if event.event_type == 'move']
        if not move_events:
            logger.warning("No movement events to generate heatmap")
            return
        
        # Extract all the X and Y positions from the movement events
        x_positions = [event.x for event in move_events]
        y_positions = [event.y for event in move_events]
        
        # Create a 2D histogram (counts how many points fall in each grid cell)
        heatmap, xedges, yedges = np.histogram2d(
            x_positions, y_positions, 
            bins=[grid_size, grid_size],
            range=[[0, resolution[0]], [0, resolution[1]]]
        )
        
        # Create the visualization
        plt.figure(figsize=(12, 8))
        plt.imshow(heatmap.T, origin='lower', cmap='hot', 
                  extent=[0, resolution[0], 0, resolution[1]])
        plt.colorbar(label='Frequency')
        plt.title('Mouse Movement Heatmap')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        
        # Save the heatmap image
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = os.path.join(self.output_dir, f"heatmap_{timestamp}.png")
        plt.savefig(fig_path)
        plt.close()
        logger.info(f"Heatmap saved to {fig_path}")

def main():
    """
    Entry point when running this script directly
    Parses command line arguments and manages the tracking process so
    it can be run from the command line with different options
    """
    parser = argparse.ArgumentParser(description='Advanced Mouse Tracker')
    parser.add_argument('--output', type=str, default='mouse_data',
                        help='Output directory for mouse data')
    parser.add_argument('--duration', type=int, default=0,
                        help='Duration to track in seconds (0 for indefinite)')
    args = parser.parse_args()
    
    # Creating and starting the tracker
    tracker = MouseTracker(output_dir=args.output)
    
    try:
        tracker.start()
        logger.info(f"Mouse tracking started. Press Ctrl+C to stop.")
        
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
        # Generate a visualization of the collected data
        tracker.generate_heatmap()
        logger.info("Mouse tracking completed")

# This section will run when the script is executed directly (not imported)
if __name__ == "__main__":
    main() 