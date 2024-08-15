#!/bin/bash

# Path to the file containing the list of packages
FILE="installed_packages.txt"
VENV_DIR="venv"

# Check if the file exists
if [ ! -f "$FILE" ]; then
    echo "$FILE does not exist."
    exit 1
fi

# Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "python3 could not be found. Please install it."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "pip could not be found. Please install it."
    exit 1
fi

# Create a virtual environment
python3 -m venv "$VENV_DIR"

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Read each line from the file and install the package
while IFS= read -r package; do
    if [ -n "$package" ]; then
        pip install "$package"
    fi
done < "$FILE"

# Deactivate the virtual environment
deactivate

echo "All packages installed in the virtual environment."