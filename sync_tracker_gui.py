#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import threading
import time
import os
import logging
import datetime
from typing import Optional, List, Tuple, Dict
from dataclasses import asdict

# Import our tracking modules
from mouse_tracker import MouseTracker
from gaze_tracker import GazeTracker, GazeEvent
from sync_tracker import SyncTracker
from mouse_analytics import MouseAnalytics

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("sync_tracker_gui.log"), logging.StreamHandler()]
)
logger = logging.getLogger("SyncTrackerGUI")

class RealTimeTrajectoryPlot:
    """Real-time plot for mouse and gaze trajectory"""
    
    def __init__(self, master, figsize=(8, 6), dpi=100):
        """Initialize the real-time plot"""
        self.master = master
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.ax.set_title("Mouse and Gaze Trajectory (Real-time)")
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")
        self.ax.grid(True, alpha=0.3)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, master)
        self.toolbar.update()
        
        # Initialize lines and points for plotting
        self.mouse_line, = self.ax.plot([], [], '-', alpha=0.5, linewidth=1, color='blue', label='Mouse')
        self.gaze_line, = self.ax.plot([], [], '-', alpha=0.5, linewidth=1, color='red', label='Gaze')
        
        self.mouse_clicks = self.ax.scatter([], [], color='green', alpha=0.7, s=30, label='Mouse clicks')
        self.gaze_fixations = self.ax.scatter([], [], color='purple', alpha=0.7, s=30, label='Gaze fixations')
        
        # Data storage
        self.mouse_x = []
        self.mouse_y = []
        self.gaze_x = []
        self.gaze_y = []
        self.click_x = []
        self.click_y = []
        self.fixation_x = []
        self.fixation_y = []
        
        # Show legend
        self.ax.legend()
        
        # Invert y-axis to match screen coordinates
        self.ax.invert_yaxis()
        
        # Draw initial empty plot
        self.canvas.draw()
        
        # Set limits
        self.set_limits()
    
    def set_limits(self, x_range: Tuple[int, int] = None, y_range: Tuple[int, int] = None):
        """Set the axis limits"""
        if x_range is None:
            # Get screen width if possible, otherwise use a default
            try:
                screen_width = self.master.winfo_screenwidth()
                x_range = (0, screen_width)
            except:
                x_range = (0, 1920)  # Default to common resolution
        
        if y_range is None:
            # Get screen height if possible, otherwise use a default
            try:
                screen_height = self.master.winfo_screenheight()
                y_range = (0, screen_height)
            except:
                y_range = (0, 1080)  # Default to common resolution
        
        self.ax.set_xlim(x_range)
        self.ax.set_ylim(y_range)
        self.canvas.draw_idle()
    
    def update_plot(self, sync_events: List[Dict]) -> None:
        """Update the plot with new synchronized events"""
        # Debug information
        logger.debug(f"Trajectory plot received {len(sync_events)} events")
        
        # Process events
        has_new_data = False
        for event in sync_events:
            if event['event_type'].startswith('mouse_'):
                if event['mouse_x'] is not None and event['mouse_y'] is not None:
                    self.mouse_x.append(event['mouse_x'])
                    self.mouse_y.append(event['mouse_y'])
                    has_new_data = True
                
                if event['event_type'] == 'mouse_click' and event.get('mouse_pressed', False):
                    self.click_x.append(event['mouse_x'])
                    self.click_y.append(event['mouse_y'])
            
            elif event['event_type'].startswith('gaze_'):
                if event['gaze_screen_x'] is not None and event['gaze_screen_y'] is not None:
                    self.gaze_x.append(event['gaze_screen_x'])
                    self.gaze_y.append(event['gaze_screen_y'])
                    has_new_data = True
                
                if event['event_type'] == 'gaze_fixation':
                    self.fixation_x.append(event['gaze_screen_x'])
                    self.fixation_y.append(event['gaze_screen_y'])
        
        # If we have new data, update the plot
        if has_new_data:
            # Limit data points to prevent slowdown
            max_points = 1000
            if len(self.mouse_x) > max_points:
                self.mouse_x = self.mouse_x[-max_points:]
                self.mouse_y = self.mouse_y[-max_points:]
            
            if len(self.gaze_x) > max_points:
                self.gaze_x = self.gaze_x[-max_points:]
                self.gaze_y = self.gaze_y[-max_points:]
            
            # Update plots
            self.mouse_line.set_data(self.mouse_x, self.mouse_y)
            self.gaze_line.set_data(self.gaze_x, self.gaze_y)
            
            self.mouse_clicks.set_offsets(np.column_stack([self.click_x, self.click_y]) if self.click_x else np.empty((0, 2)))
            self.gaze_fixations.set_offsets(np.column_stack([self.fixation_x, self.fixation_y]) if self.fixation_x else np.empty((0, 2)))
            
            # Log data points
            logger.debug(f"Mouse points: {len(self.mouse_x)}, Gaze points: {len(self.gaze_x)}")
            
            # Update axis limits if needed
            # Get min/max values with some padding
            if self.mouse_x or self.gaze_x:
                all_x = self.mouse_x + self.gaze_x
                all_y = self.mouse_y + self.gaze_y
                
                if all_x and all_y:
                    min_x, max_x = min(all_x), max(all_x)
                    min_y, max_y = min(all_y), max(all_y)
                    
                    # Add 10% padding
                    x_padding = (max_x - min_x) * 0.1
                    y_padding = (max_y - min_y) * 0.1
                    
                    self.ax.set_xlim(min_x - x_padding, max_x + x_padding)
                    self.ax.set_ylim(max_y + y_padding, min_y - y_padding)  # Inverted for screen coordinates
            
            # Redraw
            self.canvas.draw_idle()
    
    def clear(self) -> None:
        """Clear the plot"""
        self.mouse_x = []
        self.mouse_y = []
        self.gaze_x = []
        self.gaze_y = []
        self.click_x = []
        self.click_y = []
        self.fixation_x = []
        self.fixation_y = []
        
        self.mouse_line.set_data([], [])
        self.gaze_line.set_data([], [])
        self.mouse_clicks.set_offsets(np.empty((0, 2)))
        self.gaze_fixations.set_offsets(np.empty((0, 2)))
        
        self.canvas.draw_idle()

class DistancePlot:
    """Plot for mouse-gaze distance"""
    
    def __init__(self, master, figsize=(8, 4), dpi=100):
        """Initialize the distance plot"""
        self.master = master
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.ax.set_title("Mouse-Gaze Distance")
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("Distance (pixels)")
        self.ax.grid(True, alpha=0.3)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, master)
        self.toolbar.update()
        
        # Initialize line for plotting
        self.distance_line, = self.ax.plot([], [], '-', alpha=0.7, linewidth=1, color='purple')
        
        # Add horizontal line for attention threshold
        self.threshold_line = self.ax.axhline(y=100, color='g', linestyle='--', alpha=0.7, label='Attention Threshold (100px)')
        self.ax.legend()
        
        # Data storage
        self.times = []
        self.distances = []
        self.start_time = None
        
        # Draw initial empty plot
        self.canvas.draw()
    
    def update_plot(self, sync_events: List[Dict]) -> None:
        """Update the plot with new synchronized events"""
        # Debug information
        logger.debug(f"Distance plot received {len(sync_events)} events")
        
        # Process events
        has_new_data = False
        for event in sync_events:
            if event['distance'] is not None:
                if self.start_time is None:
                    self.start_time = event['timestamp']
                
                rel_time = event['timestamp'] - self.start_time
                self.times.append(rel_time)
                self.distances.append(event['distance'])
                has_new_data = True
        
        # If we have new data, update the plot
        if has_new_data:
            # Limit data points to prevent slowdown
            max_points = 1000
            if len(self.times) > max_points:
                self.times = self.times[-max_points:]
                self.distances = self.distances[-max_points:]
            
            # Update plot
            self.distance_line.set_data(self.times, self.distances)
            
            # Log data points
            logger.debug(f"Distance points: {len(self.times)}")
            
            # Update axis limits if needed
            if self.times:
                time_range = max(self.times) - min(self.times)
                max_time = max(self.times)
                
                # Set x-axis to show at least the last 10 seconds, with some padding
                self.ax.set_xlim(max(0, max_time - 10), max_time + 1)
                
                # Set y-axis to include all values with some padding
                if self.distances:
                    min_dist = max(0, min(self.distances) * 0.9)  # Don't go below 0
                    max_dist = max(100, max(self.distances) * 1.1)  # Don't go below 100px (threshold line)
                    self.ax.set_ylim(min_dist, max_dist)
            
            # Redraw
            self.canvas.draw_idle()
    
    def clear(self) -> None:
        """Clear the plot"""
        self.times = []
        self.distances = []
        self.start_time = None
        
        self.distance_line.set_data([], [])
        
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 200)
        
        self.canvas.draw_idle()

class SyncTrackerGUI:
    """GUI for synchronized mouse and gaze tracking application"""
    
    def __init__(self, root):
        """Initialize the GUI"""
        self.root = root
        root.title("Synchronized Mouse and Gaze Tracker")
        root.geometry("1280x900")
        
        # Set up the tracker
        self.data_dir = os.path.join(os.getcwd(), "mouse_data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Default gaze mode
        self.gaze_mode = "dummy"  # Options: "webcam", "tobii", "dummy"
        
        # Create the sync tracker
        self.tracker = SyncTracker(output_dir=self.data_dir, gaze_mode=self.gaze_mode)
        
        # Tracking state
        self.is_tracking = False
        self.update_interval = 500  # ms
        self.last_processed_event_idx = 0
        
        # Create the menu
        self.create_menu()
        
        # Create the main layout
        self.create_layout()
        
        # Start the UI update loop
        self.update_stats()
        
        logger.info("SyncTrackerGUI initialized")
    
    def create_menu(self):
        """Create the application menu"""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Start Tracking", command=self.start_tracking)
        file_menu.add_command(label="Stop Tracking", command=self.stop_tracking)
        file_menu.add_separator()
        file_menu.add_command(label="Generate Report", command=self.generate_report)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Options menu
        options_menu = tk.Menu(menubar, tearoff=0)
        options_menu.add_command(label="Settings", command=self.show_settings)
        menubar.add_cascade(label="Options", menu=options_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def create_layout(self):
        """Create the main application layout"""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create top frame for controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=5)
        
        # Add control buttons
        self.start_button = ttk.Button(control_frame, text="Start Tracking", command=self.start_tracking)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Tracking", command=self.stop_tracking, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        self.report_button = ttk.Button(control_frame, text="Generate Report", command=self.generate_report)
        self.report_button.pack(side=tk.LEFT, padx=5)
        
        # Add gaze mode selection
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Label(control_frame, text="Gaze Mode:").pack(side=tk.LEFT, padx=5)
        
        self.gaze_mode_var = tk.StringVar(value=self.gaze_mode)
        gaze_mode_combo = ttk.Combobox(control_frame, textvariable=self.gaze_mode_var, 
                                        values=["webcam", "tobii", "dummy"],
                                        width=10, state="readonly")
        gaze_mode_combo.pack(side=tk.LEFT, padx=5)
        gaze_mode_combo.bind("<<ComboboxSelected>>", self.on_gaze_mode_change)
        
        # Add status indicator
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Label(control_frame, text="Status:").pack(side=tk.LEFT, padx=5)
        self.status_var = tk.StringVar(value="Not Tracking")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var, font=("TkDefaultFont", 10, "bold"))
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Create stats frame
        stats_frame = ttk.LabelFrame(main_frame, text="Statistics", padding="10")
        stats_frame.pack(fill=tk.X, pady=5)
        
        # Add stats labels
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X)
        
        # Row 1
        ttk.Label(stats_grid, text="Duration:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.duration_var = tk.StringVar(value="00:00:00")
        ttk.Label(stats_grid, textvariable=self.duration_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(stats_grid, text="Mouse Events:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.mouse_events_var = tk.StringVar(value="0")
        ttk.Label(stats_grid, textvariable=self.mouse_events_var).grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(stats_grid, text="Gaze Events:").grid(row=0, column=4, sticky=tk.W, padx=5, pady=2)
        self.gaze_events_var = tk.StringVar(value="0")
        ttk.Label(stats_grid, textvariable=self.gaze_events_var).grid(row=0, column=5, sticky=tk.W, padx=5, pady=2)
        
        # Row 2
        ttk.Label(stats_grid, text="Avg Distance:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.avg_distance_var = tk.StringVar(value="0 px")
        ttk.Label(stats_grid, textvariable=self.avg_distance_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(stats_grid, text="Max Distance:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        self.max_distance_var = tk.StringVar(value="0 px")
        ttk.Label(stats_grid, textvariable=self.max_distance_var).grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(stats_grid, text="Attention Match:").grid(row=1, column=4, sticky=tk.W, padx=5, pady=2)
        self.attention_var = tk.StringVar(value="0%")
        ttk.Label(stats_grid, textvariable=self.attention_var).grid(row=1, column=5, sticky=tk.W, padx=5, pady=2)
        
        # Create notebook for visualizations
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Trajectory tab
        trajectory_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(trajectory_frame, text="Trajectories")
        
        # Add real-time trajectory plot
        self.trajectory_plot = RealTimeTrajectoryPlot(trajectory_frame)
        
        # Distance tab
        distance_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(distance_frame, text="Distance")
        
        # Add distance plot
        self.distance_plot = DistancePlot(distance_frame)
        
        # Heatmap tab (will be populated when we have data)
        self.heatmap_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.heatmap_frame, text="Heatmap")
        
        # Add placeholder text
        ttk.Label(self.heatmap_frame, text="Heatmap will be generated after tracking stops").pack(expand=True)
    
    def on_gaze_mode_change(self, event):
        """Handle gaze mode change"""
        new_mode = self.gaze_mode_var.get()
        if new_mode != self.gaze_mode:
            self.gaze_mode = new_mode
            # Re-create the tracker with new mode
            self.tracker = SyncTracker(output_dir=self.data_dir, gaze_mode=self.gaze_mode)
            logger.info(f"Gaze mode changed to {self.gaze_mode}")
    
    def start_tracking(self):
        """Start synchronized tracking"""
        if self.is_tracking:
            logger.warning("Tracking already started")
            return
        
        try:
            # Start the tracker
            success = self.tracker.start()
            if not success:
                messagebox.showerror("Error", "Failed to start tracking. Check logs for details.")
                return
                
            self.is_tracking = True
            
            # Update UI
            self.status_var.set("Tracking")
            self.status_label.config(foreground="green")
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            # Clear the plots
            self.trajectory_plot.clear()
            self.distance_plot.clear()
            
            # Reset event counter
            self.last_processed_event_idx = 0
            
            # Log
            logger.info("Synchronized tracking started")
            
        except Exception as e:
            logger.error(f"Error starting tracking: {e}")
            messagebox.showerror("Error", f"Failed to start tracking: {e}")
    
    def stop_tracking(self):
        """Stop synchronized tracking"""
        if not self.is_tracking:
            logger.warning("Tracking not started")
            return
        
        try:
            # Stop the tracker
            self.tracker.stop()
            self.is_tracking = False
            
            # Update UI
            self.status_var.set("Not Tracking")
            self.status_label.config(foreground="black")
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            
            # Log
            logger.info("Synchronized tracking stopped")
            
            # Ask if user wants to generate a report
            if messagebox.askyesno("Generate Report", "Do you want to generate a report of the tracking session?"):
                self.generate_report()
            
        except Exception as e:
            logger.error(f"Error stopping tracking: {e}")
            messagebox.showerror("Error", f"Failed to stop tracking: {e}")
    
    def update_stats(self):
        """Update statistics and visualizations"""
        if self.is_tracking and hasattr(self.tracker, 'sync_events'):
            # Calculate duration
            duration_secs = time.time() - self.tracker.start_time if hasattr(self.tracker, 'start_time') else 0
            hours, remainder = divmod(int(duration_secs), 3600)
            minutes, seconds = divmod(remainder, 60)
            self.duration_var.set(f"{hours:02}:{minutes:02}:{seconds:02}")
            
            # Count events by type
            sync_events = self.tracker.sync_events
            
            # Count mouse events
            mouse_events = sum(1 for e in sync_events if e.event_type.startswith('mouse_'))
            self.mouse_events_var.set(str(mouse_events))
            
            # Count gaze events
            gaze_events = sum(1 for e in sync_events if e.event_type.startswith('gaze_'))
            self.gaze_events_var.set(str(gaze_events))
            
            # Update metrics
            self.avg_distance_var.set(f"{self.tracker.avg_distance:.1f} px")
            self.max_distance_var.set(f"{self.tracker.max_distance:.1f} px")
            self.attention_var.set(f"{self.tracker.attention_match_percent:.1f}%")
            
            # Update plots with new events
            event_count = len(sync_events)
            if event_count > self.last_processed_event_idx:
                try:
                    # Convert sync events to dictionaries manually to avoid asdict errors
                    new_events = []
                    for e in sync_events[self.last_processed_event_idx:]:
                        # Convert each event to a dict manually
                        event_dict = {
                            'timestamp': e.timestamp,
                            'event_type': e.event_type,
                            'mouse_x': e.mouse_x,
                            'mouse_y': e.mouse_y,
                            'gaze_x': e.gaze_x,
                            'gaze_y': e.gaze_y,
                            'gaze_screen_x': e.gaze_screen_x,
                            'gaze_screen_y': e.gaze_screen_y,
                            'mouse_button': e.mouse_button,
                            'mouse_pressed': e.mouse_pressed,
                            'mouse_dx': e.mouse_dx,
                            'mouse_dy': e.mouse_dy,
                            'gaze_duration': e.gaze_duration,
                            'gaze_pupil_size': e.gaze_pupil_size,
                            'gaze_confidence': e.gaze_confidence,
                            'distance': e.distance,
                            'normalized_distance': e.normalized_distance
                        }
                        new_events.append(event_dict)
                    
                    # Update visualization
                    if new_events:
                        logger.debug(f"Processing {len(new_events)} events for plotting")
                        self.trajectory_plot.update_plot(new_events)
                        self.distance_plot.update_plot(new_events)
                    
                    # Update processed event counter
                    self.last_processed_event_idx = event_count
                except Exception as e:
                    logger.error(f"Error updating plots: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
        
        # Schedule the next update
        self.root.after(self.update_interval, self.update_stats)
    
    def generate_report(self):
        """Generate synchronized tracking report"""
        try:
            # Generate report using SyncTracker
            report_path = self.tracker.generate_report()
            
            if report_path:
                # Show success message with option to open
                if messagebox.askyesno("Report Generated", 
                                       f"Report generated successfully at:\n{report_path}\n\nDo you want to open it now?"):
                    # Open the HTML file in default browser
                    import webbrowser
                    webbrowser.open(f"file://{os.path.abspath(report_path)}")
            else:
                messagebox.showinfo("No Data", "No tracking data available to generate report")
                
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            messagebox.showerror("Error", f"Failed to generate report: {e}")
    
    def show_settings(self):
        """Show settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("500x400")
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        # Create settings form
        frame = ttk.Frame(settings_window, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Data directory setting
        ttk.Label(frame, text="Data Directory:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        dir_frame = ttk.Frame(frame)
        dir_frame.grid(row=0, column=1, sticky=tk.W+tk.E, pady=5)
        
        dir_var = tk.StringVar(value=self.data_dir)
        dir_entry = ttk.Entry(dir_frame, textvariable=dir_var, width=30)
        dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        def select_dir():
            selected_dir = filedialog.askdirectory(initialdir=self.data_dir)
            if selected_dir:
                dir_var.set(selected_dir)
        
        browse_button = ttk.Button(dir_frame, text="Browse...", command=select_dir)
        browse_button.pack(side=tk.RIGHT, padx=5)
        
        # Update interval setting
        ttk.Label(frame, text="Update Interval (ms):").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        interval_var = tk.IntVar(value=self.update_interval)
        interval_spinbox = ttk.Spinbox(frame, from_=100, to=2000, increment=100, textvariable=interval_var)
        interval_spinbox.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Gaze tracker settings
        ttk.Label(frame, text="Gaze Tracking Mode:").grid(row=2, column=0, sticky=tk.W, pady=5)
        
        mode_var = tk.StringVar(value=self.gaze_mode)
        mode_combo = ttk.Combobox(frame, textvariable=mode_var, 
                                 values=["webcam", "tobii", "dummy"],
                                 width=15, state="readonly")
        mode_combo.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Webcam settings (only visible when webcam mode selected)
        webcam_frame = ttk.LabelFrame(frame, text="Webcam Settings", padding=10)
        webcam_frame.grid(row=3, column=0, columnspan=2, sticky=tk.W+tk.E, pady=10)
        
        ttk.Label(webcam_frame, text="Camera ID:").grid(row=0, column=0, sticky=tk.W, pady=5)
        camera_id_var = tk.IntVar(value=0)
        camera_spinbox = ttk.Spinbox(webcam_frame, from_=0, to=10, increment=1, textvariable=camera_id_var, width=5)
        camera_spinbox.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        def update_webcam_frame_visibility(*args):
            if mode_var.get() == "webcam":
                webcam_frame.grid()
            else:
                webcam_frame.grid_remove()
        
        # Call once to set initial state
        update_webcam_frame_visibility()
        
        # Update when mode changes
        mode_var.trace("w", update_webcam_frame_visibility)
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=20)
        
        def save_settings():
            # Update settings
            new_dir = dir_var.get()
            if new_dir != self.data_dir:
                # Ensure directory exists
                os.makedirs(new_dir, exist_ok=True)
                self.data_dir = new_dir
                if hasattr(self.tracker, 'output_dir'):
                    self.tracker.output_dir = new_dir
            
            # Update interval
            self.update_interval = interval_var.get()
            
            # Update gaze mode if needed
            new_mode = mode_var.get()
            if new_mode != self.gaze_mode:
                self.gaze_mode = new_mode
                self.gaze_mode_var.set(new_mode)
                
                # Re-create tracker with new mode
                if not self.is_tracking:
                    self.tracker = SyncTracker(output_dir=self.data_dir, gaze_mode=self.gaze_mode)
                    logger.info(f"Recreated tracker with gaze mode: {self.gaze_mode}")
                else:
                    messagebox.showinfo("Info", "Gaze mode will be applied next time you start tracking")
            
            settings_window.destroy()
            messagebox.showinfo("Settings", "Settings saved successfully")
        
        ttk.Button(button_frame, text="Save", command=save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=settings_window.destroy).pack(side=tk.LEFT, padx=5)
    
    def show_about(self):
        """Show about dialog"""
        about_window = tk.Toplevel(self.root)
        about_window.title("About Sync Tracker")
        about_window.geometry("450x350")
        about_window.transient(self.root)
        about_window.grab_set()
        
        frame = ttk.Frame(about_window, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Synchronized Mouse and Gaze Tracker", font=("TkDefaultFont", 14, "bold")).pack(pady=10)
        ttk.Label(frame, text="Version 1.0").pack()
        ttk.Label(frame, text="A comprehensive tool for analyzing mouse and gaze patterns").pack(pady=10)
        
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        ttk.Label(frame, text="Features:").pack(anchor=tk.W)
        features = """• Synchronized mouse and gaze tracking
• Real-time visualization
• Interactive data analysis
• Attention pattern matching
• Multiple gaze tracking modes (webcam, Tobii, dummy)
• Comprehensive reporting"""
        ttk.Label(frame, text=features).pack(anchor=tk.W, padx=20)
        
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(frame, text="© 2023 ESTIA Gaze Project").pack()
        
        ttk.Button(frame, text="Close", command=about_window.destroy).pack(pady=10)

def main():
    """Run the synchronized tracker GUI application"""
    root = tk.Tk()
    app = SyncTrackerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 