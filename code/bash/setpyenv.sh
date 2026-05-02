#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to the top directory (two levels up from the script)
TOP_DIR="$SCRIPT_DIR/../.."

# Check if myenv exists
if [ ! -d "$TOP_DIR/myenv" ]; then
    echo "Creating Python virtual environment in $TOP_DIR/myenv..."
    cd "$TOP_DIR" || exit 1
    python3 -m venv myenv
    source myenv/bin/activate
    pip install --upgrade pip
    pip install opencv-python numpy matplotlib scikit-image
    deactivate
    echo "Virtual environment and dependencies installed."
else
    echo "Virtual environment already exists in $TOP_DIR/myenv."
fi

# Activate the environment
source "$TOP_DIR/myenv/bin/activate"
echo "Virtual environment activated. You can now run your Python scripts."
