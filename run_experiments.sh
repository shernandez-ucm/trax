#!/bin/bash

# Activate the virtual environment
source /home/sergio/code/tf/bin/activate

# Change to the directory containing the Python script
cd /home/sergio/code/trax/

for i in 32 64 128
do
    # Run the Python script
    python beijing_multi_site.py $i 
done