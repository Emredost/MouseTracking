#!/bin/bash

# Create a virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Set environment variables
echo "Setting environment variables..."
export MOUSE_TRACKER_DATA_DIR="$(pwd)/mouse_data"
export MOUSE_TRACKER_DEBUG=false

# Print success message
echo ""
echo "Environment setup complete!"
echo "Virtual environment has been created and dependencies installed."
echo ""
echo "Environment variables set:"
echo "MOUSE_TRACKER_DATA_DIR: $MOUSE_TRACKER_DATA_DIR"
echo "MOUSE_TRACKER_DEBUG: $MOUSE_TRACKER_DEBUG"
echo ""
echo "To activate this environment in the future, run:"
echo "source venv/bin/activate"
echo ""
echo "To run the mouse tracker GUI, run:"
echo "python mouse_tracker_gui.py" 