#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🔍 REAL DATA DEBUGGING SCRIPT

This script analyzes the characteristics of real Tobii data to understand
why the classification algorithm is biased toward saccade detection.
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

def analyze_real_gaze_data():
    """Collect and analyze real Tobii gaze data characteristics"""
    print("🔍 ANALYZING REAL TOBII DATA CHARACTERISTICS")
    print("=" * 60)
    
    try:
        from sync_tracker import SyncTracker
        
        # Create and start tracker
        tracker = SyncTracker(gaze_mode='tobii_consumer')
        
        print("🚀 Starting Tobii tracker...")
        success = tracker.start()
        
        if not success:
            print("❌ Failed to start tracker")
            return
        
        print("👀 Collecting data for 5 seconds...")
        print("   Please look steadily at this text (simulating fixation)")
        time.sleep(5)
        
        print("🔄 Now look around the screen (simulating saccades)")
        time.sleep(3)
        
        print("📊 Analyzing collected data...")
        
        # Get the gaze events
        if hasattr(tracker, 'gaze_tracker') and tracker.gaze_tracker and hasattr(tracker.gaze_tracker, 'events'):
            events = tracker.gaze_tracker.events
            
            if len(events) < 10:
                print(f"⚠️ Only {len(events)} events collected - may not be enough for analysis")
                tracker.stop()
                return
            
            print(f"✅ Analyzing {len(events)} real gaze events")
            
            # Extract coordinates and timestamps
            timestamps = [e.timestamp for e in events]
            x_coords = []
            y_coords = []
            
            for event in events:
                if hasattr(event, 'screen_x') and event.screen_x is not None:
                    x_coords.append(event.screen_x)
                    y_coords.append(event.screen_y)
                else:
                    # Convert normalized to pixels (assuming 1920x1080)
                    x_coords.append(event.x * 1920)
                    y_coords.append(event.y * 1080)
            
            if len(x_coords) < 10:
                print("⚠️ Not enough coordinate data")
                tracker.stop()
                return
            
            # Calculate characteristics
            analyze_data_characteristics(timestamps, x_coords, y_coords)
            
        tracker.stop()
        
    except Exception as e:
        print(f"❌ Error: {e}")

def analyze_data_characteristics(timestamps, x_coords, y_coords):
    """Analyze the characteristics of the collected data"""
    
    # Convert to numpy arrays
    timestamps = np.array(timestamps)
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    
    print("\n📊 DATA CHARACTERISTICS ANALYSIS")
    print("-" * 40)
    
    # Basic statistics
    duration = timestamps[-1] - timestamps[0]
    print(f"Duration: {duration:.2f} seconds")
    print(f"Events: {len(timestamps)}")
    print(f"Sampling rate: {len(timestamps)/duration:.1f} Hz")
    
    # Coordinate ranges
    print(f"\nCoordinate ranges:")
    print(f"  X: {np.min(x_coords):.1f} - {np.max(x_coords):.1f} pixels")
    print(f"  Y: {np.min(y_coords):.1f} - {np.max(y_coords):.1f} pixels")
    
    # Calculate dispersion (spatial spread)
    dispersion = np.sqrt((np.max(x_coords) - np.min(x_coords))**2 + 
                        (np.max(y_coords) - np.min(y_coords))**2)
    print(f"\nSpatial dispersion: {dispersion:.1f} pixels")
    
    # Calculate velocities
    velocities = []
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i-1]
        dx = x_coords[i] - x_coords[i-1]
        dy = y_coords[i] - y_coords[i-1]
        if dt > 0:
            velocity = np.sqrt(dx**2 + dy**2) / dt
            velocities.append(velocity)
    
    velocities = np.array(velocities)
    
    if len(velocities) > 0:
        print(f"\nVelocity statistics:")
        print(f"  Average: {np.mean(velocities):.1f} pixels/second")
        print(f"  Maximum: {np.max(velocities):.1f} pixels/second")
        print(f"  Std Dev: {np.std(velocities):.1f} pixels/second")
        print(f"  Median: {np.median(velocities):.1f} pixels/second")
    
    # Calculate accelerations
    accelerations = []
    for i in range(1, len(velocities)):
        dt = timestamps[i+1] - timestamps[i] if i+1 < len(timestamps) else 0.1
        if dt > 0:
            accel = abs(velocities[i] - velocities[i-1]) / dt
            accelerations.append(accel)
    
    accelerations = np.array(accelerations)
    
    if len(accelerations) > 0:
        print(f"\nAcceleration statistics:")
        print(f"  Average: {np.mean(accelerations):.1f} pixels/second²")
        print(f"  Maximum: {np.max(accelerations):.1f} pixels/second²")
    
    # Compare with algorithm thresholds
    print(f"\n🎯 ALGORITHM THRESHOLD COMPARISON")
    print("-" * 40)
    
    print("Current thresholds:")
    print(f"  Fixation dispersion < 35 pixels")
    print(f"  Fixation velocity < 80 pixels/second")
    print(f"  Saccade velocity > 250 pixels/second")
    print(f"  Saccade acceleration > 500 pixels/second²")
    print(f"  Pursuit velocity: 60-180 pixels/second")
    
    print(f"\nYour data vs thresholds:")
    print(f"  Dispersion: {dispersion:.1f} ({'✅ FIXATION' if dispersion < 35 else '❌ NOT FIXATION'})")
    print(f"  Avg velocity: {np.mean(velocities):.1f} ({'✅ FIXATION' if np.mean(velocities) < 80 else '🔄 PURSUIT' if 60 <= np.mean(velocities) <= 180 else '⚡ SACCADE'})")
    print(f"  Max velocity: {np.max(velocities):.1f} ({'⚡ SACCADE' if np.max(velocities) > 250 else '✅ Normal'})")
    if len(accelerations) > 0:
        print(f"  Avg acceleration: {np.mean(accelerations):.1f} ({'⚡ SACCADE' if np.mean(accelerations) > 500 else '✅ Normal'})")
    
    # Diagnosis
    print(f"\n🔧 DIAGNOSIS:")
    issues = []
    
    if dispersion > 35:
        issues.append(f"High dispersion ({dispersion:.1f} > 35) - real eyes move more than expected")
    
    if np.mean(velocities) > 80:
        issues.append(f"High average velocity ({np.mean(velocities):.1f} > 80) - real eye movements faster than threshold")
    
    if np.max(velocities) > 250:
        issues.append(f"High peak velocity ({np.max(velocities):.1f} > 250) - triggers saccade detection")
    
    if len(accelerations) > 0 and np.mean(accelerations) > 500:
        issues.append(f"High acceleration ({np.mean(accelerations):.1f} > 500) - triggers saccade detection")
    
    if issues:
        print("  Issues found:")
        for issue in issues:
            print(f"    • {issue}")
    else:
        print("  No obvious threshold issues found")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if dispersion > 35:
        new_fixation_threshold = min(100, dispersion * 1.2)
        print(f"  • Increase fixation dispersion threshold to {new_fixation_threshold:.0f} pixels")
    
    if np.mean(velocities) > 80:
        new_velocity_threshold = min(150, np.mean(velocities) * 1.5)
        print(f"  • Increase fixation velocity threshold to {new_velocity_threshold:.0f} pixels/second")
    
    if np.max(velocities) > 250:
        new_saccade_threshold = np.max(velocities) * 1.2
        print(f"  • Increase saccade velocity threshold to {new_saccade_threshold:.0f} pixels/second")
    
    return {
        'dispersion': dispersion,
        'avg_velocity': np.mean(velocities),
        'max_velocity': np.max(velocities),
        'velocity_std': np.std(velocities),
        'avg_acceleration': np.mean(accelerations) if len(accelerations) > 0 else 0,
        'duration': duration,
        'sample_rate': len(timestamps)/duration
    }

def main():
    """Run real data analysis"""
    print("This script will collect real Tobii data and analyze its characteristics")
    print("to help diagnose why the validation algorithm is biased toward saccades.\n")
    
    input("Press Enter to start data collection...")
    
    analyze_real_gaze_data()
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Review the threshold comparison above")
    print("2. Apply the recommended threshold adjustments")
    print("3. Re-run validation to test improvements")
    print("4. Use suggested values to update the classification algorithm")

if __name__ == "__main__":
    main() 