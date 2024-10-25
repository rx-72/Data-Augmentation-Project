#!/bin/bash

# Displays message to show environment is starting to run script
echo "Setting up the environement and running the script..."

# Install requirements
pip install -r requirements.txt

# Run python script
python Showcasing_Data_Augmentation_and_Generation.py > /dev/null 2>&1

# Displays message when script is finished
echo "Script execution completed!"
