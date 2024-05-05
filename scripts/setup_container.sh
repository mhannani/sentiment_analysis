echo "Create container"
container_name="huggingface+transformers-pytorch-gpu+latest"
container_name_updated="${container_name}_sentiment_analysis"

srun --mem=40G --container-image=/enroot/$container_name.sqsh --container-save=/netscratch/$USER/$container_name_updated.sqsh --container-mounts=/dev/fuse:/dev/fuse,/netscratch/$USER:/netscratch/$USER,"`pwd`":"`pwd`" --container-workdir="`pwd`"   bash -c 'bash ./image_requirements.sh'