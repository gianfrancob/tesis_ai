#! /bin/bash
CONTAINER="docker run -it --rm --runtime=nvidia --ipc=host --net=host -v ~/workspace/ml_gpu_docker_files/:/ds hamelsmu/ml-gpu"
echo 'Starting container with commmand: '$CONTAINER
eval $CONTAINER
