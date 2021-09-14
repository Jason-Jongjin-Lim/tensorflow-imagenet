#!/bin/bash

let total_gpus=${workers}*${gpus}
let gpus_per_node=${gpus}

mpirun --allow-run-as-root \
    -np $total_gpus \
    -npernode $gpus_per_node \
    --bind-to none \
    --mca btl_tcp_if_include eth0 \
    --mca orte_keep_fqdn_hostnames t \
    -x HOROVOD_TIMELINE \
    -x HOROVOD_FUSION_THRESHOLD \
    -x NCCL_SOCKET_IFNAME=eth0 \
    -x PATH \
    -x LD_LIBRARY_PATH \
    /data/tensorflow-imagenet/config_arena.sh
