#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import datetime
import json
import os
import threading
import csv
import logging
import argparse
from typing import Dict, List, Tuple, Any, Optional
from pynput import mouse
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict

# Get environment variables
DEFAULT_DATA_DIR = os.environ.get('MOUSE_TRACKER_DATA_DIR', os.path.join(os.getcwd(), "mouse_data"))
DEBUG_MODE = os.environ.get('MOUSE_TRACKER_DEBUG', 'false').lower() == 'true'

# Set up logging
log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("mouse_tracker.log"), logging.StreamHandler()]
)
logger = logging.getLogger("MouseTracker")

@dataclass
class MouseEvent:
    """Data class for storing mouse events"""
    timestamp: float
    event_type: str  # 'move', 'click', 'scroll'
    x: int = 0
    y: int = 0
    button: Optional[str] = None
    pressed: Optional[bool] = None
    dx: Optional[int] = None
    dy: Optional[int] = None

class MouseTracker:
    """Advanced mouse tracking class"""
    
    def __init__(self, output_dir: str = DEFAULT_DATA_DIR):
        """Initialize the mouse tracker"""
        self.events: List[MouseEvent] = []
        self.running = False
        self.output_dir = output_dir
        self.listener = None
        self.last_event_time = 0
        self.lock = threading.Lock()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Track metrics
        self.total_distance = 0
        self.last_position = (0, 0)
        self.start_time = 0
        
        logger.info("MouseTracker initialized")
        logger.debug(f"Using output directory: {self.output_dir}")
        logger.debug(f"Debug mode: {DEBUG_MODE}")
    
    def on_move(self, x: int, y: int) -> None:
        """Callback for mouse movement"""
        timestamp = time.time()
        
        # Calculate metrics if we have a previous position
        if self.last_position != (0, 0):
            distance = np.sqrt((x - self.last_position[0])**2 + (y - self.last_position[1])**2)
            self.total_distance += distance
        
        self.last_position = (x, y)
        
        # Create and store event
        with self.lock:
            self.events.append(MouseEvent(
                timestamp=timestamp,
                event_type='move',
                x=x,
                y=y
            ))
        
        # Throttle logging to reduce output noise
        if timestamp - self.last_event_time > 1.0:
            logger.debug(f"Mouse moved to {x}, {y}")
            self.last_event_time = timestamp
    
    def on_click(self, x: int, y: int, button: mouse.Button, pressed: bool) -> None:
        """Callback for mouse clicks"""
        button_name = str(button).replace('Button.', '')
        timestamp = time.time()
        
        with self.lock:
            self.events.append(MouseEvent(
                timestamp=timestamp,
                event_type='click',
                x=x,
                y=y,
                button=button_name,
                pressed=pressed
            ))
        
        logger.info(f"Mouse {'pressed' if pressed else 'released'} {button_name} at {x}, {y}")
    
    def on_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        """Callback for mouse scroll"""
        timestamp = time.time()
        
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
    
    def start(self) -> None:
        """Start tracking mouse events"""
        if self.running:
            logger.warning("Tracking already started")
            return
        
        self.running = True
        self.start_time = time.time()
        logger.info("Starting mouse tracking")
        
        # Start the listener in a non-blocking way
        self.listener = mouse.Listener(
            on_move=self.on_move,
            on_click=self.on_click,
            on_scroll=self.on_scroll
        )
        self.listener.start()
        
        # Start the periodic save thread
        self.save_thread = threading.Thread(target=self._periodic_save)
        self.save_thread.daemon = True
        self.save_thread.start()
    
    def stop(self) -> None:
        """Stop tracking mouse events"""
        if not self.running:
            logger.warning("Tracking not started")
            return
        
        self.running = False
        logger.info("Stopping mouse tracking")
        
        if self.listener:
            self.listener.stop()
            self.listener.join()
        
        # Save data on stop
        self.save_data()
    
    def _periodic_save(self) -> None:
        """Periodically save data to avoid data loss"""
        while self.running:
            time.sleep(60)  # Save every minute
            if self.running:
                self.save_data(periodic=True)
    
    def save_data(self, periodic: bool = False) -> None:
        """Save tracking data to files"""
        if not self.events:
            logger.warning("No events to save")
            return
        
        # Create timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_suffix = "periodic" if periodic else "final"
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, f"mouse_events_{timestamp}_{file_suffix}.csv")
        
        with self.lock:
            events_to_save = self.events.copy()
            if not periodic:
                self.events = []
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'event_type', 'x', 'y', 'button', 'pressed', 'dx', 'dy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for event in events_to_save:
                writer.writerow(asdict(event))
        
        # Also save to JSON for easier analysis
        json_path = os.path.join(self.output_dir, f"mouse_events_{timestamp}_{file_suffix}.json")
        with open(json_path, 'w') as jsonfile:
            json.dump([asdict(event) for event in events_to_save], jsonfile, indent=2)
        
        duration = time.time() - self.start_time
        logger.info(f"Saved {len(events_to_save)} events to {csv_path} and {json_path}")
        logger.info(f"Total tracking duration: {duration:.2f} seconds")
        logger.info(f"Total distance moved: {self.total_distance:.2f} pixels")

    def generate_heatmap(self, resolution: Tuple[int, int] = (1920, 1080), grid_size: int = 50) -> None:
        """Generate a heatmap of mouse positions"""
        if not self.events:
            logger.warning("No events to generate heatmap")
            return
        
        # Extract move events
        move_events = [event for event in self.events if event.event_type == 'move']
        if not move_events:
            logger.warning("No movement events to generate heatmap")
            return
        
        # Create a 2D histogram
        x_positions = [event.x for event in move_events]
        y_positions = [event.y for event in move_events]
        
        heatmap, xedges, yedges = np.histogram2d(
            x_positions, y_positions, 
            bins=[grid_size, grid_size],
            range=[[0, resolution[0]], [0, resolution[1]]]
        )
        
        # Plot the heatmap
        plt.figure(figsize=(12, 8))
        plt.imshow(heatmap.T, origin='lower', cmap='hot', 
                  extent=[0, resolution[0], 0, resolution[1]])
        plt.colorbar(label='Frequency')
        plt.title('Mouse Movement Heatmap')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        
        # Save the figure
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = os.path.join(self.output_dir, f"heatmap_{timestamp}.png")
        plt.savefig(fig_path)
        plt.close()
        logger.info(f"Heatmap saved to {fig_path}")

def main():
    """Main function to run the mouse tracker"""
    parser = argparse.ArgumentParser(description='Advanced Mouse Tracker')
    parser.add_argument('--output', type=str, default='mouse_data',
                        help='Output directory for mouse data')
    parser.add_argument('--duration', type=int, default=0,
                        help='Duration to track in seconds (0 for indefinite)')
    args = parser.parse_args()
    
    tracker = MouseTracker(output_dir=args.output)
    
    try:
        tracker.start()
        logger.info(f"Mouse tracking started. Press Ctrl+C to stop.")
        
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
        # Generate visualization
        tracker.generate_heatmap()
        logger.info("Mouse tracking completed")

if __name__ == "__main__":
    main() 