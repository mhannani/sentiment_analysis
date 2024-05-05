echo "Running container"
container_name="huggingface+transformers-pytorch-gpu+latest_sentiment_analysis"

srun --container-image=/netscratch/$USER/$container_name.sqsh --container-save=/netscratch/$USER/$container_name.sqsh --pty /bin/bash