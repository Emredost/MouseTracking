#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🧪 Gaze Classification Validation System Runner
Run this to start the validation experiment
"""

import os
import sys
import logging

def setup_logging():
    """Setup logging for the validation system"""
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join("logs", "validation.log")),
            logging.StreamHandler()
        ]
    )

def main():
    """Main entry point"""
    print("🧪 GAZE CLASSIFICATION VALIDATION SYSTEM")
    print("=" * 50)
    print("🚀 Starting validation application...")
    
    setup_logging()
    
    try:
        # Import and run the validation GUI
        from gaze_validation_standalone import ValidationExperimentGUI
        import tkinter as tk
        
        root = tk.Tk()
        app = ValidationExperimentGUI(root)
        
        print("✅ Validation system ready!")
        print("📋 Follow the GUI instructions to run experiments")
        
        root.mainloop()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("📁 Make sure gaze_validation_standalone.py is in the same directory")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.error(f"Validation system error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 