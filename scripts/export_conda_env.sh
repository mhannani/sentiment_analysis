#!/bin/bash

# Define the name of the conda environment
ENV_NAME="sentiment_analysis"

# Initialize Conda
eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate "$ENV_NAME"

# Export the environment to YAML
conda env export  --no-builds --from-history | grep -v "prefix" > environment.yml

# Deactivate the conda environment
conda deactivate