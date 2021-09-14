#!/bin/bash

arena submit mpi \
 --name imagenet-tensorflow \
 --gpus=4 \
 --workers=2 \
 --data=imagenet-data:/data \
 --working-dir=/data \
 --image=cuda11-registry.cn-beijing.cr.aliyuncs.com/aiacc/cuda11:v3.0 "sh /data/tensorflow-imagenet/launch_arena.sh"
