#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any, Optional
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("mouse_analytics.log"), logging.StreamHandler()]
)
logger = logging.getLogger("MouseAnalytics")

class MouseAnalytics:
    """Advanced analytics for mouse tracking data"""
    
    def __init__(self, data_dir: str = "mouse_data"):
        """Initialize the analytics engine"""
        self.data_dir = data_dir
        self.data = None
        self.click_data = None
        self.move_data = None
        self.scroll_data = None
        logger.info(f"MouseAnalytics initialized with data directory: {data_dir}")
    
    def load_data(self, file_pattern: str = "*.json") -> None:
        """Load data from JSON files in the data directory"""
        all_files = glob.glob(os.path.join(self.data_dir, file_pattern))
        
        if not all_files:
            logger.error(f"No files found matching pattern {file_pattern} in {self.data_dir}")
            return
        
        all_events = []
        
        for file_path in all_files:
            try:
                with open(file_path, 'r') as f:
                    events = json.load(f)
                    all_events.extend(events)
                    logger.info(f"Loaded {len(events)} events from {file_path}")
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")
        
        if not all_events:
            logger.error("No events loaded from files")
            return
        
        # Convert to pandas DataFrame for easier analysis
        self.data = pd.DataFrame(all_events)
        
        # Convert timestamp to datetime for easier analysis
        self.data['datetime'] = pd.to_datetime(self.data['timestamp'], unit='s')
        
        # Split by event type
        self.move_data = self.data[self.data['event_type'] == 'move']
        self.click_data = self.data[self.data['event_type'] == 'click']
        self.scroll_data = self.data[self.data['event_type'] == 'scroll']
        
        logger.info(f"Loaded {len(self.data)} total events")
        logger.info(f"Move events: {len(self.move_data)}")
        logger.info(f"Click events: {len(self.click_data)}")
        logger.info(f"Scroll events: {len(self.scroll_data)}")
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate key metrics from the tracking data"""
        if self.data is None or len(self.data) == 0:
            logger.error("No data loaded to calculate metrics")
            return {}
        
        metrics = {}
        
        # Time metrics
        if len(self.data) > 0:
            start_time = self.data['timestamp'].min()
            end_time = self.data['timestamp'].max()
            duration = end_time - start_time
            metrics['duration_seconds'] = duration
            metrics['duration_formatted'] = str(pd.Timedelta(seconds=duration))
            metrics['start_time'] = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
            metrics['end_time'] = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
        
        # Movement metrics
        if len(self.move_data) > 1:
            # Calculate distances between consecutive points
            x_diff = self.move_data['x'].diff().dropna()
            y_diff = self.move_data['y'].diff().dropna()
            
            # Calculate Euclidean distances
            distances = np.sqrt(x_diff**2 + y_diff**2)
            
            metrics['total_distance'] = distances.sum()
            metrics['average_distance_per_movement'] = distances.mean()
            metrics['max_distance_single_movement'] = distances.max()
            
            # Calculate speeds (pixels per second)
            time_diffs = self.move_data['timestamp'].diff().dropna()
            speeds = distances / np.maximum(time_diffs, 1e-6)  # Avoid division by zero
            
            metrics['average_speed'] = speeds.mean()
            metrics['max_speed'] = speeds.max()
            metrics['median_speed'] = speeds.median()
            
            # Movement heatmap data
            metrics['x_range'] = (self.move_data['x'].min(), self.move_data['x'].max())
            metrics['y_range'] = (self.move_data['y'].min(), self.move_data['y'].max())
        
        # Click metrics
        if len(self.click_data) > 0:
            # Count clicks by type and state
            press_data = self.click_data[self.click_data['pressed'] == True]
            metrics['total_clicks'] = len(press_data)
            
            # Count by button
            button_counts = press_data['button'].value_counts().to_dict()
            metrics['button_counts'] = button_counts
            
            # Calculate time between clicks
            if len(press_data) > 1:
                click_times = press_data.sort_values('timestamp')
                time_between_clicks = click_times['timestamp'].diff().dropna()
                
                metrics['avg_time_between_clicks'] = time_between_clicks.mean()
                metrics['median_time_between_clicks'] = time_between_clicks.median()
        
        # Scroll metrics
        if len(self.scroll_data) > 0:
            metrics['total_scrolls'] = len(self.scroll_data)
            metrics['avg_scroll_dx'] = self.scroll_data['dx'].mean()
            metrics['avg_scroll_dy'] = self.scroll_data['dy'].mean()
        
        logger.info(f"Calculated {len(metrics)} metrics")
        return metrics
    
    def generate_trajectory_plot(self, output_file: str = None) -> None:
        """Generate a trajectory plot of mouse movements"""
        if self.move_data is None or len(self.move_data) == 0:
            logger.error("No movement data to generate trajectory plot")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Get a sample of the data if there are too many points
        if len(self.move_data) > 10000:
            move_sample = self.move_data.sample(10000)
            logger.info(f"Sampled 10000 points from {len(self.move_data)} for trajectory plot")
        else:
            move_sample = self.move_data
        
        # Plot the trajectory
        plt.plot(move_sample['x'], move_sample['y'], '-', alpha=0.5, linewidth=1)
        
        # Add click points if available
        if self.click_data is not None and len(self.click_data) > 0:
            click_press = self.click_data[self.click_data['pressed'] == True]
            
            # Different colors for different buttons
            button_colors = {
                'left': 'red',
                'right': 'blue',
                'middle': 'green'
            }
            
            for button, color in button_colors.items():
                button_data = click_press[click_press['button'] == button]
                if len(button_data) > 0:
                    plt.scatter(button_data['x'], button_data['y'], 
                               color=color, alpha=0.7, s=50, label=f'{button} click')
        
        plt.title('Mouse Movement Trajectory')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Invert y-axis to match screen coordinates
        plt.gca().invert_yaxis()
        
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Trajectory plot saved to {output_file}")
        else:
            output_file = os.path.join(self.data_dir, f"trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(output_file)
            logger.info(f"Trajectory plot saved to {output_file}")
        
        plt.close()
    
    def generate_heatmap(self, grid_size: int = 50, output_file: str = None) -> None:
        """Generate a heatmap of mouse positions"""
        if self.move_data is None or len(self.move_data) == 0:
            logger.error("No movement data to generate heatmap")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Get max screen dimensions
        x_max = self.move_data['x'].max()
        y_max = self.move_data['y'].max()
        
        # Create 2D histogram
        heatmap, xedges, yedges = np.histogram2d(
            self.move_data['x'], self.move_data['y'],
            bins=[grid_size, grid_size],
            range=[[0, x_max], [0, y_max]]
        )
        
        # Apply logarithmic scaling for better visualization
        heatmap = np.log1p(heatmap)
        
        # Plot the heatmap
        plt.imshow(heatmap.T, origin='lower', cmap='hot',
                  extent=[0, x_max, 0, y_max],
                  aspect='auto')
        
        plt.colorbar(label='Log(frequency)')
        plt.title('Mouse Movement Heatmap')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        
        # Invert y-axis to match screen coordinates
        plt.gca().invert_yaxis()
        
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Heatmap saved to {output_file}")
        else:
            output_file = os.path.join(self.data_dir, f"heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(output_file)
            logger.info(f"Heatmap saved to {output_file}")
        
        plt.close()
    
    def generate_speed_plot(self, output_file: str = None) -> None:
        """Generate a plot of mouse movement speed over time"""
        if self.move_data is None or len(self.move_data) < 2:
            logger.error("Insufficient movement data to generate speed plot")
            return
        
        # Calculate speeds
        x_diff = self.move_data['x'].diff().dropna()
        y_diff = self.move_data['y'].diff().dropna()
        time_diff = self.move_data['timestamp'].diff().dropna()
        
        # Calculate Euclidean distances
        distances = np.sqrt(x_diff**2 + y_diff**2)
        
        # Calculate speeds (pixels per second)
        speeds = distances / np.maximum(time_diff, 1e-6)  # Avoid division by zero
        
        # Create a DataFrame for plotting
        speed_df = pd.DataFrame({
            'timestamp': self.move_data['timestamp'].iloc[1:].values,
            'datetime': self.move_data['datetime'].iloc[1:].values,
            'speed': speeds.values
        })
        
        # Remove extreme outliers (speeds > 3 std devs from mean)
        speed_mean = speed_df['speed'].mean()
        speed_std = speed_df['speed'].std()
        speed_df = speed_df[speed_df['speed'] <= speed_mean + 3*speed_std]
        
        plt.figure(figsize=(14, 6))
        
        # Create the plot
        plt.plot(speed_df['datetime'], speed_df['speed'], '-', alpha=0.5, linewidth=1)
        
        # Add a smoothed trend line
        window_size = min(len(speed_df) // 10, 100)
        if window_size > 0:
            speed_df['smooth_speed'] = speed_df['speed'].rolling(window=window_size, center=True).mean()
            plt.plot(speed_df['datetime'], speed_df['smooth_speed'], 'r-', linewidth=2, label='Smoothed speed')
        
        plt.title('Mouse Movement Speed Over Time')
        plt.xlabel('Time')
        plt.ylabel('Speed (pixels/second)')
        plt.grid(True, alpha=0.3)
        
        if len(speed_df) > 0:
            plt.legend()
        
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Speed plot saved to {output_file}")
        else:
            output_file = os.path.join(self.data_dir, f"speed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(output_file)
            logger.info(f"Speed plot saved to {output_file}")
        
        plt.close()
    
    def generate_click_distribution(self, output_file: str = None) -> None:
        """Generate a plot showing the distribution of mouse clicks"""
        if self.click_data is None or len(self.click_data) == 0:
            logger.error("No click data to generate distribution")
            return
        
        # Filter for press events only
        press_data = self.click_data[self.click_data['pressed'] == True]
        
        if len(press_data) == 0:
            logger.error("No press events to analyze")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Count clicks by button type
        button_counts = press_data['button'].value_counts()
        
        # Plot the distribution
        ax = sns.barplot(x=button_counts.index, y=button_counts.values)
        
        # Add value labels on top of each bar
        for i, v in enumerate(button_counts.values):
            ax.text(i, v + 0.1, str(v), ha='center')
        
        plt.title('Mouse Click Distribution by Button')
        plt.xlabel('Button')
        plt.ylabel('Count')
        
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Click distribution saved to {output_file}")
        else:
            output_file = os.path.join(self.data_dir, f"click_dist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(output_file)
            logger.info(f"Click distribution saved to {output_file}")
        
        plt.close()
    
    def generate_report(self, output_file: str = None) -> None:
        """Generate a comprehensive HTML report of mouse activity"""
        if self.data is None or len(self.data) == 0:
            logger.error("No data to generate report")
            return
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Generate plots
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        trajectory_file = os.path.join(self.data_dir, f"trajectory_{timestamp}.png")
        heatmap_file = os.path.join(self.data_dir, f"heatmap_{timestamp}.png")
        speed_file = os.path.join(self.data_dir, f"speed_{timestamp}.png")
        click_dist_file = os.path.join(self.data_dir, f"click_dist_{timestamp}.png")
        
        self.generate_trajectory_plot(trajectory_file)
        self.generate_heatmap(output_file=heatmap_file)
        self.generate_speed_plot(output_file=speed_file)
        
        if len(self.click_data) > 0:
            self.generate_click_distribution(output_file=click_dist_file)
        
        # Create HTML report
        if not output_file:
            output_file = os.path.join(self.data_dir, f"mouse_report_{timestamp}.html")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Mouse Tracking Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
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
            <h1>Mouse Tracking Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Summary</h2>
                <div class="metric">
                    <span class="metric-name">Time Period:</span>
                    <span class="metric-value">{metrics.get('start_time', 'N/A')} to {metrics.get('duration_formatted', 'N/A')}</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Duration:</span>
                    <span class="metric-value">{metrics.get('duration_formatted', 'N/A')}</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Total Events:</span>
                    <span class="metric-value">{len(self.data)}</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Movement Events:</span>
                    <span class="metric-value">{len(self.move_data)}</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Click Events:</span>
                    <span class="metric-value">{len(self.click_data) if self.click_data is not None else 'N/A'}</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Scroll Events:</span>
                    <span class="metric-value">{len(self.scroll_data) if self.scroll_data is not None else 'N/A'}</span>
                </div>
            </div>
            
            <div class="section">
                <h2>Movement Metrics</h2>
                <div class="metric">
                    <span class="metric-name">Total Distance:</span>
                    <span class="metric-value">{metrics.get('total_distance', 'N/A'):.2f} pixels</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Average Speed:</span>
                    <span class="metric-value">{metrics.get('average_speed', 'N/A'):.2f} pixels/second</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Maximum Speed:</span>
                    <span class="metric-value">{metrics.get('max_speed', 'N/A'):.2f} pixels/second</span>
                </div>
            </div>
            
            <div class="section">
                <h2>Click Metrics</h2>
                <div class="metric">
                    <span class="metric-name">Total Clicks:</span>
                    <span class="metric-value">{metrics.get('total_clicks', 'N/A')}</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Average Time Between Clicks:</span>
                    <span class="metric-value">{metrics.get('avg_time_between_clicks', 'N/A'):.2f} seconds</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Click Distribution:</span>
                </div>
                <table>
                    <tr>
                        <th>Button</th>
                        <th>Count</th>
                    </tr>
        """
        
        # Add button counts table rows
        button_counts = metrics.get('button_counts', {})
        for button, count in button_counts.items():
            html_content += f"""
                    <tr>
                        <td>{button}</td>
                        <td>{count}</td>
                    </tr>
            """
        
        html_content += f"""
                </table>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                
                <h3>Mouse Movement Trajectory</h3>
                <img src="{os.path.basename(trajectory_file)}" alt="Mouse Movement Trajectory">
                
                <h3>Mouse Movement Heatmap</h3>
                <img src="{os.path.basename(heatmap_file)}" alt="Mouse Movement Heatmap">
                
                <h3>Mouse Speed Over Time</h3>
                <img src="{os.path.basename(speed_file)}" alt="Mouse Speed Over Time">
        """
        
        if len(self.click_data) > 0:
            html_content += f"""
                <h3>Mouse Click Distribution</h3>
                <img src="{os.path.basename(click_dist_file)}" alt="Mouse Click Distribution">
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML report at {output_file}")
        return output_file

def main():
    """Main function to run analytics"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Mouse Tracking Analytics')
    parser.add_argument('--data-dir', type=str, default='mouse_data',
                        help='Directory containing mouse tracking data')
    parser.add_argument('--report', action='store_true',
                        help='Generate a comprehensive HTML report')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path for the report')
    
    args = parser.parse_args()
    
    analytics = MouseAnalytics(data_dir=args.data_dir)
    analytics.load_data()
    
    if args.report:
        analytics.generate_report(output_file=args.output)
    else:
        # Calculate and display metrics
        metrics = analytics.calculate_metrics()
        print("\nMouse Tracking Metrics:")
        print("=======================")
        
        for key, value in metrics.items():
            print(f"{key}: {value}")
        
        # Generate individual visualizations
        analytics.generate_trajectory_plot()
        analytics.generate_heatmap()
        analytics.generate_speed_plot()
        
        if analytics.click_data is not None and len(analytics.click_data) > 0:
            analytics.generate_click_distribution()

if __name__ == "__main__":
    main() 