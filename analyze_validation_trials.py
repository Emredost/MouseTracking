#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🔍 VALIDATION TRIAL DATA ANALYZER

This script analyzes the specific characteristics of the gaze data
collected during validation trials to understand why fixation and 
pursuit are still being misclassified as saccades.
"""

import sys
import os
import time
import numpy as np

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

def collect_trial_data():
    """Collect data for each trial type to analyze specific characteristics"""
    print("🔍 ANALYZING VALIDATION TRIAL DATA")
    print("=" * 60)
    print("This will collect data for each trial type to see exactly")
    print("what the classification algorithm is seeing.")
    print()
    
    try:
        from sync_tracker import SyncTracker
        
        # Create and start tracker
        tracker = SyncTracker(gaze_mode='tobii_consumer')
        
        print("🚀 Starting Tobii tracker...")
        success = tracker.start()
        
        if not success:
            print("❌ Failed to start tracker")
            return
        
        # Analyze each trial type
        trial_types = [
            ("fixation", "👁️ FIXATION: Look steadily at this text (don't move your eyes)"),
            ("pursuit", "🔄 PURSUIT: Slowly move your eyes left and right following this → ← "),  
            ("saccade", "⚡ SACCADE: Quickly jump your eyes between these two points: • --- •")
        ]
        
        results = {}
        
        for trial_type, instruction in trial_types:
            print(f"\n{instruction}")
            input("Press Enter when ready to start this trial...")
            
            # Clear previous events
            if hasattr(tracker.gaze_tracker, 'events'):
                initial_count = len(tracker.gaze_tracker.events)
            
            print(f"👀 Collecting {trial_type} data for 3 seconds...")
            start_time = time.time()
            
            # Collect for 3 seconds
            while time.time() - start_time < 3.0:
                time.sleep(0.05)  # 20Hz sampling
            
            # Get the trial data
            if hasattr(tracker.gaze_tracker, 'events'):
                all_events = tracker.gaze_tracker.events
                trial_events = [e for e in all_events if e.timestamp >= start_time]
                
                if len(trial_events) >= 3:
                    # Analyze this trial's data
                    characteristics = analyze_trial_characteristics(trial_events, trial_type)
                    results[trial_type] = characteristics
                    print(f"✅ Collected {len(trial_events)} events for {trial_type}")
                else:
                    print(f"⚠️ Only {len(trial_events)} events for {trial_type}")
            
            time.sleep(1)  # Brief pause between trials
        
        tracker.stop()
        
        # Compare results
        compare_trial_characteristics(results)
        
    except Exception as e:
        print(f"❌ Error: {e}")

def analyze_trial_characteristics(gaze_events, trial_type):
    """Analyze characteristics of a specific trial"""
    
    # Extract coordinates and timestamps
    timestamps = [e.timestamp for e in gaze_events]
    x_coords = []
    y_coords = []
    
    for event in gaze_events:
        if hasattr(event, 'screen_x') and event.screen_x is not None:
            x_coords.append(event.screen_x)
            y_coords.append(event.screen_y)
        else:
            # Convert normalized to pixels
            x_coords.append(event.x * 1920)
            y_coords.append(event.y * 1080)
    
    # Convert to numpy arrays
    timestamps = np.array(timestamps)
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    
    # Calculate all the metrics the algorithm uses
    duration = (timestamps[-1] - timestamps[0]) * 1000  # milliseconds
    
    # Dispersion
    dispersion = np.sqrt((np.max(x_coords) - np.min(x_coords))**2 + 
                        (np.max(y_coords) - np.min(y_coords))**2)
    
    # Velocities
    velocities = []
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i-1]
        dx = x_coords[i] - x_coords[i-1]
        dy = y_coords[i] - y_coords[i-1]
        if dt > 0:
            velocity = np.sqrt(dx**2 + dy**2) / dt
            velocities.append(velocity)
    
    velocities = np.array(velocities)
    avg_velocity = np.mean(velocities) if len(velocities) > 0 else 0
    max_velocity = np.max(velocities) if len(velocities) > 0 else 0
    velocity_std = np.std(velocities) if len(velocities) > 1 else 0
    
    # Accelerations
    accelerations = []
    for i in range(1, len(velocities)):
        dt = timestamps[i+1] - timestamps[i] if i+1 < len(timestamps) else 0.1
        if dt > 0:
            accel = abs(velocities[i] - velocities[i-1]) / dt
            accelerations.append(accel)
    
    accelerations = np.array(accelerations)
    avg_acceleration = np.mean(accelerations) if len(accelerations) > 0 else 0
    
    return {
        'trial_type': trial_type,
        'duration': duration,
        'dispersion': dispersion,
        'avg_velocity': avg_velocity,
        'max_velocity': max_velocity,
        'velocity_std': velocity_std,
        'avg_acceleration': avg_acceleration,
        'event_count': len(gaze_events)
    }

def compare_trial_characteristics(results):
    """Compare characteristics across trial types and recommend threshold adjustments"""
    
    print(f"\n🔬 DETAILED TRIAL ANALYSIS")
    print("=" * 80)
    
    # Current algorithm thresholds (updated ones)
    current_thresholds = {
        'fixation_dispersion': 120,
        'fixation_velocity': 150,
        'fixation_velocity_std': 200,
        'saccade_max_velocity': 1000,
        'saccade_avg_velocity': 400,
        'saccade_acceleration': 2000,
        'pursuit_dispersion': 100,
        'pursuit_velocity_min': 150,
        'pursuit_velocity_max': 600,
        'pursuit_velocity_std': 400
    }
    
    print("📊 CURRENT THRESHOLDS:")
    for key, value in current_thresholds.items():
        print(f"  {key}: {value}")
    
    print(f"\n📋 TRIAL DATA COMPARISON:")
    print(f"{'Metric':<20} {'Fixation':<12} {'Pursuit':<12} {'Saccade':<12} {'Threshold':<15} {'Issue?'}")
    print("-" * 85)
    
    metrics = [
        ('Duration (ms)', 'duration', None),
        ('Dispersion (px)', 'dispersion', None),
        ('Avg Velocity', 'avg_velocity', None),
        ('Max Velocity', 'max_velocity', None),
        ('Velocity Std', 'velocity_std', None),
        ('Avg Accel', 'avg_acceleration', None),
        ('Event Count', 'event_count', None)
    ]
    
    issues_found = []
    recommendations = []
    
    for metric_name, metric_key, threshold_key in metrics:
        fixation_val = results.get('fixation', {}).get(metric_key, 0)
        pursuit_val = results.get('pursuit', {}).get(metric_key, 0)
        saccade_val = results.get('saccade', {}).get(metric_key, 0)
        
        threshold_str = "N/A"
        issue = ""
        
        # Check specific threshold violations
        if metric_key == 'dispersion':
            if fixation_val > current_thresholds['fixation_dispersion']:
                issue = "❌ FIX"
                issues_found.append(f"Fixation dispersion too high: {fixation_val:.1f} > {current_thresholds['fixation_dispersion']}")
                recommendations.append(f"Increase fixation_dispersion to {fixation_val * 1.2:.0f}")
            
            if pursuit_val < current_thresholds['pursuit_dispersion']:
                issue += " ❌ PUR"
                issues_found.append(f"Pursuit dispersion too low: {pursuit_val:.1f} < {current_thresholds['pursuit_dispersion']}")
                recommendations.append(f"Decrease pursuit_dispersion to {pursuit_val * 0.8:.0f}")
            
            threshold_str = f"Fix<{current_thresholds['fixation_dispersion']}, Pur>{current_thresholds['pursuit_dispersion']}"
        
        elif metric_key == 'avg_velocity':
            if fixation_val > current_thresholds['fixation_velocity']:
                issue = "❌ FIX"
                issues_found.append(f"Fixation velocity too high: {fixation_val:.1f} > {current_thresholds['fixation_velocity']}")
                recommendations.append(f"Increase fixation_velocity to {fixation_val * 1.3:.0f}")
            
            if not (current_thresholds['pursuit_velocity_min'] <= pursuit_val <= current_thresholds['pursuit_velocity_max']):
                issue += " ❌ PUR"
                issues_found.append(f"Pursuit velocity outside range: {pursuit_val:.1f} not in {current_thresholds['pursuit_velocity_min']}-{current_thresholds['pursuit_velocity_max']}")
            
            if saccade_val < current_thresholds['saccade_avg_velocity']:
                issue += " ❌ SAC"
                issues_found.append(f"Saccade velocity too low: {saccade_val:.1f} < {current_thresholds['saccade_avg_velocity']}")
                recommendations.append(f"Decrease saccade_avg_velocity to {saccade_val * 0.8:.0f}")
            
            threshold_str = f"Fix<{current_thresholds['fixation_velocity']}, Sac>{current_thresholds['saccade_avg_velocity']}"
        
        elif metric_key == 'max_velocity':
            if saccade_val < current_thresholds['saccade_max_velocity']:
                issue = "❌ SAC"
                issues_found.append(f"Saccade max velocity too low: {saccade_val:.1f} < {current_thresholds['saccade_max_velocity']}")
                recommendations.append(f"Decrease saccade_max_velocity to {saccade_val * 0.8:.0f}")
            
            threshold_str = f"Sac>{current_thresholds['saccade_max_velocity']}"
        
        elif metric_key == 'velocity_std':
            if fixation_val > current_thresholds['fixation_velocity_std']:
                issue = "❌ FIX"
                issues_found.append(f"Fixation velocity std too high: {fixation_val:.1f} > {current_thresholds['fixation_velocity_std']}")
                recommendations.append(f"Increase fixation_velocity_std to {fixation_val * 1.2:.0f}")
            
            if pursuit_val > current_thresholds['pursuit_velocity_std']:
                issue += " ❌ PUR"
                issues_found.append(f"Pursuit velocity std too high: {pursuit_val:.1f} > {current_thresholds['pursuit_velocity_std']}")
                recommendations.append(f"Increase pursuit_velocity_std to {pursuit_val * 1.2:.0f}")
            
            threshold_str = f"Fix<{current_thresholds['fixation_velocity_std']}, Pur<{current_thresholds['pursuit_velocity_std']}"
        
        elif metric_key == 'avg_acceleration':
            if saccade_val < current_thresholds['saccade_acceleration']:
                issue = "❌ SAC"
                issues_found.append(f"Saccade acceleration too low: {saccade_val:.1f} < {current_thresholds['saccade_acceleration']}")
                recommendations.append(f"Decrease saccade_acceleration to {saccade_val * 0.8:.0f}")
            
            threshold_str = f"Sac>{current_thresholds['saccade_acceleration']}"
        
        print(f"{metric_name:<20} {fixation_val:<12.1f} {pursuit_val:<12.1f} {saccade_val:<12.1f} {threshold_str:<15} {issue}")
    
    print(f"\n🚨 ISSUES IDENTIFIED:")
    if issues_found:
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")
    else:
        print("  No obvious threshold issues found")
    
    print(f"\n💡 RECOMMENDED THRESHOLD ADJUSTMENTS:")
    if recommendations:
        unique_recs = list(set(recommendations))
        for i, rec in enumerate(unique_recs, 1):
            print(f"  {i}. {rec}")
    else:
        print("  No specific adjustments recommended")
    
    # Generate new threshold values
    if results:
        print(f"\n🔧 SUGGESTED NEW THRESHOLDS (based on your data):")
        
        fixation_data = results.get('fixation', {})
        pursuit_data = results.get('pursuit', {})  
        saccade_data = results.get('saccade', {})
        
        if fixation_data:
            print(f"  # Fixation thresholds")
            print(f"  fixation_dispersion = {fixation_data.get('dispersion', 120) * 1.3:.0f}  # was 120")
            print(f"  fixation_velocity = {fixation_data.get('avg_velocity', 150) * 1.4:.0f}  # was 150")
            print(f"  fixation_velocity_std = {fixation_data.get('velocity_std', 200) * 1.3:.0f}  # was 200")
        
        if saccade_data:
            print(f"  # Saccade thresholds")
            print(f"  saccade_max_velocity = {saccade_data.get('max_velocity', 1000) * 0.7:.0f}  # was 1000")
            print(f"  saccade_avg_velocity = {saccade_data.get('avg_velocity', 400) * 0.7:.0f}  # was 400")
            print(f"  saccade_acceleration = {saccade_data.get('avg_acceleration', 2000) * 0.7:.0f}  # was 2000")
        
        if pursuit_data:
            print(f"  # Pursuit thresholds")
            print(f"  pursuit_dispersion = {pursuit_data.get('dispersion', 100) * 0.8:.0f}  # was 100")
            pursuit_vel = pursuit_data.get('avg_velocity', 300)
            print(f"  pursuit_velocity_range = ({pursuit_vel * 0.7:.0f}, {pursuit_vel * 1.5:.0f})  # was (150, 600)")
            print(f"  pursuit_velocity_std = {pursuit_data.get('velocity_std', 400) * 1.3:.0f}  # was 400")

def main():
    """Run trial data analysis"""
    print("This script will help you perform each type of eye movement")
    print("and analyze exactly what the algorithm sees for each one.")
    print("This will help us calibrate the thresholds precisely.\n")
    
    input("Press Enter to start the analysis...")
    
    collect_trial_data()

if __name__ == "__main__":
    main() 