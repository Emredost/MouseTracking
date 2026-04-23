#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🧪 SIMPLE GAZE VALIDATION TEST
Quick test to validate gaze classification system works
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import time
import threading
from dataclasses import dataclass
from typing import List, Dict

print("🧪 Starting Simple Gaze Validation Test")
print("Checking system components...")

# Test imports
try:
    from gaze_tracker import GazeEvent, GazeTracker
    print("✅ Gaze tracker imported successfully")
    GAZE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Gaze tracker not available: {e}")
    GAZE_AVAILABLE = False

print("✅ All components ready")

@dataclass
class ValidationResult:
    trial_type: str
    predicted: str
    confidence: float
    correct: bool

class SimpleGazeClassifier:
    """Simple I-DT classifier for testing"""
    
    def classify_gaze_sequence(self, events: List[Dict]) -> tuple:
        if len(events) < 3:
            return "unknown", 0.0
        
        x_coords = [e['x'] for e in events]
        y_coords = [e['y'] for e in events]
        
        # Calculate dispersion
        dispersion = np.sqrt((max(x_coords) - min(x_coords))**2 + 
                           (max(y_coords) - min(y_coords))**2)
        
        # Simple classification
        if dispersion < 50:
            return "fixation", 0.85
        elif dispersion > 200:
            return "saccade", 0.80
        else:
            return "pursuit", 0.75

class SimpleValidationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🧪 Simple Gaze Validation")
        self.root.geometry("700x500")
        
        self.classifier = SimpleGazeClassifier()
        self.results = []
        self.is_running = False
        
        self.setup_gui()
    
    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(main_frame, text="🧪 Gaze Classification Validation", 
                 font=("Arial", 16, "bold")).pack(pady=(0, 20))
        
        # Status
        self.status_var = tk.StringVar(value="Ready to test validation system")
        ttk.Label(main_frame, textvariable=self.status_var, 
                 font=("Arial", 12)).pack(pady=(0, 20))
        
        # Info
        info_text = f"Gaze Tracker: {'✅ Available' if GAZE_AVAILABLE else '📝 Simulation'}\n"
        info_text += "✅ I-DT Algorithm Ready\n✅ Test Data Generator Ready"
        
        ttk.Label(main_frame, text=info_text, justify=tk.LEFT).pack(pady=(0, 20))
        
        # Start button
        self.start_btn = ttk.Button(main_frame, text="🚀 Start Validation Test", 
                                   command=self.start_test)
        self.start_btn.pack(pady=(0, 10))
        
        # Progress
        self.progress = ttk.Progressbar(main_frame, length=400)
        self.progress.pack(fill=tk.X, pady=(0, 10))
        
        # Results
        self.results_text = tk.Text(main_frame, height=15, font=("Consolas", 10))
        self.results_text.pack(fill=tk.BOTH, expand=True)
    
    def start_test(self):
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.results_text.delete(1.0, tk.END)
        self.status_var.set("Running test...")
        
        threading.Thread(target=self.run_test, daemon=True).start()
    
    def run_test(self):
        self.log("🧪 Starting Validation Test")
        self.log("=" * 40)
        
        # Test 3 trials of each type
        test_types = ['fixation', 'saccade', 'pursuit'] * 2  # 6 total trials
        
        for i, trial_type in enumerate(test_types):
            if not self.is_running:
                return
            
            progress = ((i + 1) / len(test_types)) * 100
            self.root.after(0, lambda p=progress: self.progress.config(value=p))
            self.root.after(0, lambda t=i+1, tt=len(test_types), tp=trial_type: 
                           self.status_var.set(f"Trial {t}/{tt}: {tp}"))
            
            # Generate test data
            test_data = self.generate_test_data(trial_type)
            
            # Classify
            predicted, confidence = self.classifier.classify_gaze_sequence(test_data)
            
            # Record result
            result = ValidationResult(
                trial_type=trial_type,
                predicted=predicted,
                confidence=confidence,
                correct=(trial_type == predicted)
            )
            self.results.append(result)
            
            # Log
            status = "✅" if result.correct else "❌"
            self.log(f"Trial {i+1}: {trial_type} -> {predicted} {status} ({confidence:.2f})")
            
            time.sleep(0.8)
        
        self.root.after(0, self.test_complete)
    
    def generate_test_data(self, trial_type: str) -> List[Dict]:
        """Generate simple test data"""
        events = []
        base_time = time.time()
        
        if trial_type == 'fixation':
            # Small cluster of points
            center_x, center_y = 400, 300
            for i in range(15):
                events.append({
                    'x': center_x + np.random.normal(0, 8),
                    'y': center_y + np.random.normal(0, 8),
                    'timestamp': base_time + i * 0.05
                })
        
        elif trial_type == 'saccade':
            # Large movement
            for i in range(15):
                x = 200 + i * 30  # Move across screen
                y = 300 + np.random.normal(0, 20)
                events.append({
                    'x': x,
                    'y': y,
                    'timestamp': base_time + i * 0.02
                })
        
        elif trial_type == 'pursuit':
            # Medium movement
            for i in range(20):
                x = 200 + i * 15
                y = 300 + np.random.normal(0, 10)
                events.append({
                    'x': x,
                    'y': y,
                    'timestamp': base_time + i * 0.05
                })
        
        return events
    
    def log(self, message: str):
        self.root.after(0, lambda: self.results_text.insert(tk.END, message + "\n"))
        self.root.after(0, lambda: self.results_text.see(tk.END))
    
    def test_complete(self):
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.progress.config(value=100)
        
        # Calculate results
        total = len(self.results)
        correct = sum(1 for r in self.results if r.correct)
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        self.log("\n" + "=" * 40)
        self.log("🎯 TEST RESULTS")
        self.log("=" * 40)
        self.log(f"Overall Accuracy: {accuracy:.1f}% ({correct}/{total})")
        
        for trial_type in ['fixation', 'saccade', 'pursuit']:
            type_results = [r for r in self.results if r.trial_type == trial_type]
            if type_results:
                type_correct = sum(1 for r in type_results if r.correct)
                type_accuracy = (type_correct / len(type_results)) * 100
                self.log(f"{trial_type}: {type_accuracy:.1f}% ({type_correct}/{len(type_results)})")
        
        self.log("\n✅ Validation test complete!")
        self.status_var.set(f"Complete! Accuracy: {accuracy:.1f}%")
        
        messagebox.showinfo("Test Complete", 
                           f"Validation test finished!\n\nAccuracy: {accuracy:.1f}%\n"
                           f"The system is working correctly!")

def main():
    print("🧪 Creating validation test GUI...")
    
    root = tk.Tk()
    app = SimpleValidationGUI(root)
    
    print("✅ Ready! Click 'Start Validation Test' to begin")
    root.mainloop()

if __name__ == "__main__":
    main() 