# Won't be executed directly
# It will be executed when runninh sh setup_container.sh

#!/bin/bash
gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys A4B469963BF863CC

# add the key
gpg --export --armor A4B469963BF863CC | apt-key add -

# updating
apt-get update

# install system packages
apt install -y curl git wget unzip

# create Downloads folder
mkdir /Downloads && cd /Downloads

# Download Miniconda
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -b -f

# Load conda initialization in current shell session
source /root/miniconda3/etc/profile.d/conda.sh

# activate conda env on startup
conda init --reverse bash
