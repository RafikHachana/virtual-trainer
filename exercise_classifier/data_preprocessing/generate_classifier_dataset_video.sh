#!/bin/bash

for (( i=0; i<=9; i++ ))
do
    python3 generate_classifier_dataset_video.py $i
done
