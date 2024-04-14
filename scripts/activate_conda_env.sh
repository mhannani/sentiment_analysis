#!/bin/bash

# Initialize Conda
eval "$(conda shell.bash hook)"

# Deactivate any currently active environment (including base)
conda deactivate

# activate the conda env
conda activate sentiment_analysis
