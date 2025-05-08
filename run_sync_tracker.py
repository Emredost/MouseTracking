#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Synchronized Mouse and Gaze Tracking Application
------------------------------------------------

This script provides an easy way to run the synchronized mouse and gaze tracker
with proper error handling and environment setup.

Usage:
    python run_sync_tracker.py [--gaze-mode {webcam|tobii|dummy}]
"""

import os
import sys
import argparse
import subprocess
import platform
import logging
import tkinter as tk
from tkinter import messagebox
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("sync_tracker_runner.log"), logging.StreamHandler()]
)
logger = logging.getLogger("Runner")

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import numpy
        import matplotlib
        import pandas
        import cv2
        import dlib
        logger.info("All dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False

def check_model_files():
    """Check if required model files exist and download them if needed"""
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    model_path = os.path.join(model_dir, "shape_predictor_68_face_landmarks.dat")
    
    if not os.path.exists(model_path):
        logger.warning(f"Face landmark model not found at {model_path}")
        
        # Create models directory
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            # Show download progress message
            root = tk.Tk()
            root.withdraw()
            proceed = messagebox.askyesno(
                "Download Required Model",
                "The eye tracking model file is missing. Do you want to download it now?\n\n"
                "This will download approximately 65MB of data."
            )
            
            if not proceed:
                logger.info("User canceled model download")
                return False
                
            # Different download methods based on platform
            if platform.system() == "Darwin":  # macOS
                logger.info("Downloading model using curl (macOS)")
                subprocess.run([
                    "curl", "-L", 
                    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
                    "-o", os.path.join(model_dir, "shape_predictor_68_face_landmarks.dat.bz2")
                ], check=True)
                
                logger.info("Extracting model file")
                subprocess.run([
                    "python", "-c", 
                    "import bz2; open('models/shape_predictor_68_face_landmarks.dat', 'wb').write(bz2.BZ2File('models/shape_predictor_68_face_landmarks.dat.bz2').read())"
                ], check=True)
                
            else:  # Windows or Linux
                logger.info("Downloading model using Python requests")
                import requests
                import bz2
                
                url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
                compressed_path = os.path.join(model_dir, "shape_predictor_68_face_landmarks.dat.bz2")
                
                with open(compressed_path, 'wb') as f:
                    response = requests.get(url, stream=True)
                    total_length = response.headers.get('content-length')
                    
                    if total_length is None:
                        f.write(response.content)
                    else:
                        dl = 0
                        total_length = int(total_length)
                        for data in response.iter_content(chunk_size=4096):
                            dl += len(data)
                            f.write(data)
                            done = int(50 * dl / total_length)
                            sys.stdout.write("\r[%s%s] %s%%" % ('=' * done, ' ' * (50-done), int(100 * dl / total_length)))
                            sys.stdout.flush()
                
                logger.info("\nExtracting model file")
                with open(model_path, 'wb') as f:
                    f.write(bz2.BZ2File(compressed_path).read())
            
            logger.info(f"Model downloaded and extracted to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            return False
    else:
        logger.info(f"Model file found at {model_path}")
        return True

def check_macos_permissions():
    """Check and guide user for macOS permissions required for tracking"""
    if platform.system() != "Darwin":
        return True  # Not on macOS, no need to check
        
    logger.info("Checking macOS permissions")
    
    # Check if we can show a dialog
    try:
        root = tk.Tk()
        root.withdraw()
        
        # Check which version of macOS we're on
        mac_version = platform.mac_ver()[0]
        logger.info(f"macOS Version: {mac_version}")
        
        major_version = int(mac_version.split('.')[0])
        
        # Ask user to set up permissions
        messagebox.showinfo(
            "macOS Permissions Required",
            "For mouse tracking to work, you need to grant Input Monitoring permission to Terminal "
            "(or your Python IDE).\n\n"
            "The System Settings window will now open to the appropriate location.\n\n"
            "Please click the '+' button and add Terminal to the allowed applications."
        )
        
        # Open the appropriate System Preferences panel based on macOS version
        if major_version >= 14:  # macOS Sequoia (14) or later
            subprocess.run(["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent"], check=False)
        else:  # Older versions
            subprocess.run(["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"], check=False)
        
        # Wait for user confirmation
        proceeded = messagebox.askyesno(
            "Permissions Confirmation",
            "Have you added Terminal to the Input Monitoring permissions?\n\n"
            "Click 'Yes' to continue, 'No' to try again."
        )
        
        if not proceeded:
            logger.warning("User did not confirm permissions setup")
            return False
            
        root.destroy()
        return True
        
    except Exception as e:
        logger.error(f"Error checking permissions: {e}")
        return True  # Continue anyway

def run_application(gaze_mode="webcam"):
    """Run the synchronized tracking application"""
    try:
        logger.info(f"Starting sync_tracker_gui.py with gaze mode: {gaze_mode}")
        
        # Set environment variable for gaze mode
        os.environ['GAZE_TRACKER_MODE'] = gaze_mode
        
        # Import and run
        from sync_tracker_gui import main
        main()
        
    except Exception as e:
        logger.error(f"Error running application: {e}")
        logger.error(traceback.format_exc())
        
        # Show error message
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Application Error",
            f"An error occurred while running the application:\n\n{str(e)}\n\n"
            f"See sync_tracker_runner.log for details."
        )
        return False
    
    return True

def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Synchronized Mouse and Gaze Tracker')
    parser.add_argument('--gaze-mode', type=str, default='webcam',
                        choices=['webcam', 'tobii', 'dummy'],
                        help='Gaze tracking mode (webcam, tobii, or dummy)')
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Missing dependencies. Please run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Check for model files
    if args.gaze_mode == 'webcam' and not check_model_files():
        logger.warning("Missing model file. Falling back to dummy mode.")
        args.gaze_mode = 'dummy'
    
    # Check macOS permissions
    if not check_macos_permissions():
        logger.warning("macOS permissions may not be correctly set")
        
        # Show warning but continue
        root = tk.Tk()
        root.withdraw()
        messagebox.showwarning(
            "Permissions Warning",
            "Mouse tracking may not work correctly without proper permissions.\n\n"
            "If tracking fails, please restart and configure permissions."
        )
    
    # Run the application
    run_application(args.gaze_mode)

if __name__ == "__main__":
    main() 