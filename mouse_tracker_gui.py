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

# Import our tracking module
from mouse_tracker import MouseTracker
from mouse_analytics import MouseAnalytics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("mouse_tracker_gui.log"), logging.StreamHandler()]
)
logger = logging.getLogger("MouseTrackerGUI")

class RealTimeTrajectoryPlot:
    """Real-time plot for mouse trajectory"""
    
    def __init__(self, master, figsize=(6, 4), dpi=100):
        """Initialize the real-time plot"""
        self.master = master
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.ax.set_title("Mouse Trajectory (Real-time)")
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")
        self.ax.grid(True, alpha=0.3)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, master)
        self.toolbar.update()
        
        # Initialize line and points for plotting
        self.trajectory_line, = self.ax.plot([], [], '-', alpha=0.5, linewidth=1, color='blue')
        self.left_clicks = self.ax.scatter([], [], color='red', alpha=0.7, s=30, label='Left clicks')
        self.right_clicks = self.ax.scatter([], [], color='blue', alpha=0.7, s=30, label='Right clicks')
        self.middle_clicks = self.ax.scatter([], [], color='green', alpha=0.7, s=30, label='Middle clicks')
        
        # Data storage
        self.x_data = []
        self.y_data = []
        self.left_click_x = []
        self.left_click_y = []
        self.right_click_x = []
        self.right_click_y = []
        self.middle_click_x = []
        self.middle_click_y = []
        
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
    
    def update_plot(self, events: List[Dict]) -> None:
        """Update the plot with new events"""
        # Process events
        for event in events:
            if event['event_type'] == 'move':
                self.x_data.append(event['x'])
                self.y_data.append(event['y'])
            elif event['event_type'] == 'click' and event.get('pressed', False):
                button = event.get('button', '')
                if button == 'left':
                    self.left_click_x.append(event['x'])
                    self.left_click_y.append(event['y'])
                elif button == 'right':
                    self.right_click_x.append(event['x'])
                    self.right_click_y.append(event['y'])
                elif button == 'middle':
                    self.middle_click_x.append(event['x'])
                    self.middle_click_y.append(event['y'])
        
        # Limit data points to prevent slowdown
        max_points = 1000
        if len(self.x_data) > max_points:
            self.x_data = self.x_data[-max_points:]
            self.y_data = self.y_data[-max_points:]
        
        # Update plots
        self.trajectory_line.set_data(self.x_data, self.y_data)
        self.left_clicks.set_offsets(np.column_stack([self.left_click_x, self.left_click_y]) if self.left_click_x else np.empty((0, 2)))
        self.right_clicks.set_offsets(np.column_stack([self.right_click_x, self.right_click_y]) if self.right_click_x else np.empty((0, 2)))
        self.middle_clicks.set_offsets(np.column_stack([self.middle_click_x, self.middle_click_y]) if self.middle_click_x else np.empty((0, 2)))
        
        # Redraw
        self.canvas.draw_idle()
    
    def clear(self) -> None:
        """Clear the plot"""
        self.x_data = []
        self.y_data = []
        self.left_click_x = []
        self.left_click_y = []
        self.right_click_x = []
        self.right_click_y = []
        self.middle_click_x = []
        self.middle_click_y = []
        
        self.trajectory_line.set_data([], [])
        self.left_clicks.set_offsets(np.empty((0, 2)))
        self.right_clicks.set_offsets(np.empty((0, 2)))
        self.middle_clicks.set_offsets(np.empty((0, 2)))
        
        self.canvas.draw_idle()

class MouseTrackerGUI:
    """GUI for mouse tracking application"""
    
    def __init__(self, root):
        """Initialize the GUI"""
        self.root = root
        root.title("Advanced Mouse Tracker")
        root.geometry("1200x800")
        
        # Set up the tracker
        self.data_dir = os.path.join(os.getcwd(), "mouse_data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.tracker = MouseTracker(output_dir=self.data_dir)
        
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
        
        logger.info("MouseTrackerGUI initialized")
    
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
        
        # Add status indicator
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Label(control_frame, text="Status:").pack(side=tk.LEFT, padx=5)
        self.status_var = tk.StringVar(value="Not Tracking")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var, font=("TkDefaultFont", 10, "bold"))
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Create stats frame at the bottom
        stats_frame = ttk.LabelFrame(main_frame, text="Statistics", padding="10")
        stats_frame.pack(fill=tk.X, pady=5)
        
        # Add stats labels
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X)
        
        # Column 1
        ttk.Label(stats_grid, text="Duration:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(stats_grid, text="Mouse Events:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(stats_grid, text="Mouse Clicks:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        
        # Column 2
        self.duration_var = tk.StringVar(value="00:00:00")
        ttk.Label(stats_grid, textvariable=self.duration_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        self.events_var = tk.StringVar(value="0")
        ttk.Label(stats_grid, textvariable=self.events_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        self.clicks_var = tk.StringVar(value="0")
        ttk.Label(stats_grid, textvariable=self.clicks_var).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Column 3
        ttk.Label(stats_grid, text="Distance:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        ttk.Label(stats_grid, text="Avg Speed:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        ttk.Label(stats_grid, text="Last Position:").grid(row=2, column=2, sticky=tk.W, padx=5, pady=2)
        
        # Column 4
        self.distance_var = tk.StringVar(value="0 px")
        ttk.Label(stats_grid, textvariable=self.distance_var).grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        
        self.speed_var = tk.StringVar(value="0 px/s")
        ttk.Label(stats_grid, textvariable=self.speed_var).grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)
        
        self.position_var = tk.StringVar(value="(0, 0)")
        ttk.Label(stats_grid, textvariable=self.position_var).grid(row=2, column=3, sticky=tk.W, padx=5, pady=2)
        
        # Create notebook for visualizations
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Trajectory tab
        trajectory_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(trajectory_frame, text="Trajectory")
        
        # Add real-time trajectory plot
        self.trajectory_plot = RealTimeTrajectoryPlot(trajectory_frame)
        
        # Heatmap tab (will be populated when we have data)
        self.heatmap_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.heatmap_frame, text="Heatmap")
        
        # Add placeholder text
        ttk.Label(self.heatmap_frame, text="Heatmap will be generated after tracking stops").pack(expand=True)
    
    def start_tracking(self):
        """Start mouse tracking"""
        if self.is_tracking:
            logger.warning("Tracking already started")
            return
        
        try:
            # Start the tracker
            self.tracker.start()
            self.is_tracking = True
            
            # Update UI
            self.status_var.set("Tracking")
            self.status_label.config(foreground="green")
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            # Clear the plot
            self.trajectory_plot.clear()
            
            # Reset event counter
            self.last_processed_event_idx = 0
            
            # Log
            logger.info("Mouse tracking started")
            
        except Exception as e:
            logger.error(f"Error starting tracking: {e}")
            messagebox.showerror("Error", f"Failed to start tracking: {e}")
    
    def stop_tracking(self):
        """Stop mouse tracking"""
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
            logger.info("Mouse tracking stopped")
            
            # Ask if user wants to generate a report
            if messagebox.askyesno("Generate Report", "Do you want to generate a report of the tracking session?"):
                self.generate_report()
            
        except Exception as e:
            logger.error(f"Error stopping tracking: {e}")
            messagebox.showerror("Error", f"Failed to stop tracking: {e}")
    
    def update_stats(self):
        """Update statistics and visualizations"""
        if self.is_tracking:
            # Calculate duration
            duration_secs = time.time() - self.tracker.start_time
            hours, remainder = divmod(int(duration_secs), 3600)
            minutes, seconds = divmod(remainder, 60)
            self.duration_var.set(f"{hours:02}:{minutes:02}:{seconds:02}")
            
            # Update events count
            event_count = len(self.tracker.events)
            self.events_var.set(str(event_count))
            
            # Update clicks count
            click_count = sum(1 for e in self.tracker.events if e.event_type == 'click' and e.pressed)
            self.clicks_var.set(str(click_count))
            
            # Update distance
            self.distance_var.set(f"{self.tracker.total_distance:.1f} px")
            
            # Calculate speed (if we have events)
            if event_count > 1:
                # Calculate average speed over last second
                now = time.time()
                recent_events = [e for e in self.tracker.events if e.event_type == 'move' and now - e.timestamp <= 1.0]
                
                if len(recent_events) >= 2:
                    first = recent_events[0]
                    last = recent_events[-1]
                    time_diff = last.timestamp - first.timestamp
                    
                    if time_diff > 0:
                        dist = np.sqrt((last.x - first.x)**2 + (last.y - first.y)**2)
                        speed = dist / time_diff
                        self.speed_var.set(f"{speed:.1f} px/s")
            
            # Update position
            if event_count > 0 and self.tracker.events[-1].event_type == 'move':
                last_x = self.tracker.events[-1].x
                last_y = self.tracker.events[-1].y
                self.position_var.set(f"({last_x}, {last_y})")
            
            # Update trajectory plot with new events
            if event_count > self.last_processed_event_idx:
                new_events = [e.__dict__ for e in self.tracker.events[self.last_processed_event_idx:]]
                self.trajectory_plot.update_plot(new_events)
                self.last_processed_event_idx = event_count
        
        # Schedule the next update
        self.root.after(self.update_interval, self.update_stats)
    
    def generate_report(self):
        """Generate analytics report"""
        try:
            # Create analytics object
            analytics = MouseAnalytics(data_dir=self.data_dir)
            analytics.load_data()
            
            # Generate report
            report_file = analytics.generate_report()
            
            if report_file:
                # Show success message with option to open
                if messagebox.askyesno("Report Generated", 
                                       f"Report generated successfully at:\n{report_file}\n\nDo you want to open it now?"):
                    # Open the HTML file in default browser
                    import webbrowser
                    webbrowser.open(f"file://{os.path.abspath(report_file)}")
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            messagebox.showerror("Error", f"Failed to generate report: {e}")
    
    def show_settings(self):
        """Show settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x300")
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
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=20)
        
        def save_settings():
            # Update settings
            new_dir = dir_var.get()
            if new_dir != self.data_dir:
                # Ensure directory exists
                os.makedirs(new_dir, exist_ok=True)
                self.data_dir = new_dir
                self.tracker.output_dir = new_dir
            
            # Update interval
            self.update_interval = interval_var.get()
            
            settings_window.destroy()
            messagebox.showinfo("Settings", "Settings saved successfully")
        
        ttk.Button(button_frame, text="Save", command=save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=settings_window.destroy).pack(side=tk.LEFT, padx=5)
    
    def show_about(self):
        """Show about dialog"""
        about_window = tk.Toplevel(self.root)
        about_window.title("About Mouse Tracker")
        about_window.geometry("400x300")
        about_window.transient(self.root)
        about_window.grab_set()
        
        frame = ttk.Frame(about_window, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Advanced Mouse Tracker", font=("TkDefaultFont", 14, "bold")).pack(pady=10)
        ttk.Label(frame, text="Version 1.0").pack()
        ttk.Label(frame, text="A sophisticated mouse tracking and analytics tool").pack(pady=10)
        
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        ttk.Label(frame, text="Features:").pack(anchor=tk.W)
        features = """• Real-time mouse tracking
• Movement visualization
• Click tracking
• Comprehensive analytics
• Data export and reporting"""
        ttk.Label(frame, text=features).pack(anchor=tk.W, padx=20)
        
        ttk.Button(frame, text="Close", command=about_window.destroy).pack(pady=10)

def main():
    """Run the mouse tracker GUI application"""
    root = tk.Tk()
    app = MouseTrackerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 