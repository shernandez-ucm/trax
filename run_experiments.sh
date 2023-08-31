#!/bin/bash

# Activate the virtual environment
source /home/sergio/code/tf/bin/activate

# Change to the directory containing the Python script
cd /home/sergio/code/trax/

for i in 32 64 128
do 
    for j in 64 128 256
    do
        for k in 1e-5 3e-5 7e-5 10e-5
        do  
            # Run the Python script
            python beijing_multi_site.py $i $j $k
        done
    done
done