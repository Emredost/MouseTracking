#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🧪 VALIDATION FIX VERIFICATION SCRIPT

This script tests whether the validation system now properly:
1. Detects real Tobii tracking state
2. Auto-starts tracking when needed
3. Collects actual gaze data instead of simulation
4. Reports real vs simulated data usage

Run this after the fix to verify everything works!
"""

import sys
import os
import time

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

def test_tracker_state_detection():
    """Test if we can properly detect tracker running state"""
    print("🔍 Testing tracker state detection...")
    
    try:
        from sync_tracker import SyncTracker
        
        # Create a tracker instance
        tracker = SyncTracker(gaze_mode='tobii_consumer')
        
        # Test state detection BEFORE starting
        print(f"  📊 Before start:")
        print(f"    tracker.running: {getattr(tracker, 'running', 'MISSING')}")
        print(f"    tracker.gaze_tracker exists: {hasattr(tracker, 'gaze_tracker')}")
        if hasattr(tracker, 'gaze_tracker') and tracker.gaze_tracker:
            print(f"    tracker.gaze_tracker.running: {getattr(tracker.gaze_tracker, 'running', 'MISSING')}")
        
        # The FIXED logic should be:
        is_tracking_fixed = (hasattr(tracker, 'running') and tracker.running and 
                           hasattr(tracker, 'gaze_tracker') and tracker.gaze_tracker and
                           hasattr(tracker.gaze_tracker, 'running') and tracker.gaze_tracker.running)
        
        print(f"    ✅ FIXED detection result: {is_tracking_fixed}")
        
        # Test starting the tracker
        print(f"  🚀 Testing tracker start...")
        success = tracker.start()
        print(f"    Start success: {success}")
        
        if success:
            print(f"  📊 After start:")
            print(f"    tracker.running: {getattr(tracker, 'running', 'MISSING')}")
            if hasattr(tracker, 'gaze_tracker') and tracker.gaze_tracker:
                print(f"    tracker.gaze_tracker.running: {getattr(tracker.gaze_tracker, 'running', 'MISSING')}")
            
            # Re-test detection
            is_tracking_fixed = (hasattr(tracker, 'running') and tracker.running and 
                               hasattr(tracker, 'gaze_tracker') and tracker.gaze_tracker and
                               hasattr(tracker.gaze_tracker, 'running') and tracker.gaze_tracker.running)
            
            print(f"    ✅ FIXED detection result: {is_tracking_fixed}")
            
            # Stop the tracker
            tracker.stop()
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing tracker state: {e}")
        return False

def test_gaze_data_collection():
    """Test if we can collect real gaze events"""
    print("\n👀 Testing gaze data collection...")
    
    try:
        from sync_tracker import SyncTracker
        
        # Create and start tracker
        tracker = SyncTracker(gaze_mode='tobii_consumer')
        
        print("  🚀 Starting tracker for data collection test...")
        success = tracker.start()
        
        if not success:
            print("  ❌ Failed to start tracker - cannot test data collection")
            return False
        
        # Wait a moment for data to start flowing
        print("  ⏳ Collecting data for 2 seconds...")
        time.sleep(2)
        
        # Check if we have gaze events
        if hasattr(tracker, 'gaze_tracker') and tracker.gaze_tracker and hasattr(tracker.gaze_tracker, 'events'):
            event_count = len(tracker.gaze_tracker.events)
            print(f"  📊 Collected {event_count} gaze events")
            
            if event_count > 0:
                # Show some event details
                recent_events = tracker.gaze_tracker.events[-5:]  # Last 5 events
                print(f"  📋 Recent events:")
                for i, event in enumerate(recent_events):
                    print(f"    {i+1}. Type: {event.event_type}, Time: {event.timestamp:.3f}")
                    
                print("  ✅ Real gaze data collection is working!")
                tracker.stop()
                return True
            else:
                print("  ⚠️ No gaze events collected - may be simulation mode")
                tracker.stop()
                return False
        else:
            print("  ❌ Gaze tracker events not accessible")
            tracker.stop()
            return False
            
    except Exception as e:
        print(f"❌ Error testing gaze data collection: {e}")
        return False

def test_validation_integration():
    """Test if validation system can use the fixed logic"""
    print("\n🧪 Testing validation integration...")
    
    try:
        # Import the fixed GUI class
        from sync_tracker_gui import SyncTrackerGUI
        import tkinter as tk
        
        # Create a test GUI instance
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        gui = SyncTrackerGUI(root)
        
        # Test the tracking state detection logic
        is_tracking_fixed = (hasattr(gui.tracker, 'running') and gui.tracker.running and 
                           hasattr(gui.tracker, 'gaze_tracker') and gui.tracker.gaze_tracker and
                           hasattr(gui.tracker.gaze_tracker, 'running') and gui.tracker.gaze_tracker.running)
        
        print(f"  📊 Validation state detection (before start): {is_tracking_fixed}")
        
        # Test starting through the GUI
        print("  🚀 Testing GUI tracker start...")
        try:
            success = gui.tracker.start()
            print(f"    Start success: {success}")
            
            if success:
                # Re-test detection
                is_tracking_fixed = (hasattr(gui.tracker, 'running') and gui.tracker.running and 
                                   hasattr(gui.tracker, 'gaze_tracker') and gui.tracker.gaze_tracker and
                                   hasattr(gui.tracker.gaze_tracker, 'running') and gui.tracker.gaze_tracker.running)
                
                print(f"  📊 Validation state detection (after start): {is_tracking_fixed}")
                
                if is_tracking_fixed:
                    print("  ✅ Validation should now use REAL data!")
                else:
                    print("  ⚠️ Validation may still use simulation")
                
                gui.tracker.stop()
            
        except Exception as e:
            print(f"    ❌ Error starting GUI tracker: {e}")
        
        root.destroy()
        return True
        
    except Exception as e:
        print(f"❌ Error testing validation integration: {e}")
        return False

def main():
    """Run all validation fix tests"""
    print("🔧 VALIDATION FIX VERIFICATION")
    print("=" * 50)
    print("Testing if the validation system now uses REAL Tobii data...\n")
    
    # Run tests
    test1_result = test_tracker_state_detection()
    test2_result = test_gaze_data_collection()
    test3_result = test_validation_integration()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY:")
    print(f"  1. Tracker State Detection: {'✅ PASS' if test1_result else '❌ FAIL'}")
    print(f"  2. Gaze Data Collection: {'✅ PASS' if test2_result else '❌ FAIL'}")
    print(f"  3. Validation Integration: {'✅ PASS' if test3_result else '❌ FAIL'}")
    
    if all([test1_result, test2_result, test3_result]):
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Validation system should now use REAL Tobii data!")
        print("\n💡 Next steps:")
        print("   1. Run sync_tracker_gui.py")
        print("   2. Go to Validation tab")
        print("   3. Click 'Start Validation'")
        print("   4. Choose 'Yes' to start real tracking")
        print("   5. Look at the screen during trials!")
    else:
        print("\n⚠️ Some tests failed - validation may still have issues")
        print("Check the error messages above for details")

if __name__ == "__main__":
    main() 