#!/bin/bash

DATA_DIR="/data/imagenet/imagenet_data/train"

python /data/tensorflow-imagenet/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
    --model=resnet50 \
    --batch_size=128 \
    --data_name=imagenet \
    --data_dir=${DATA_DIR} \
    --use_fp16=True \
    --xla=True \
    --num_warmup_batches=500 \
    --num_batches=1500 \
    --display_every=100 \
    --num_gpus=1 \
    --horovod_device=gpu \
    --variable_update=perseus \
    --batch_group_size=4 \
    --optimizer=momentum \
    --momentum=0.9 \
    --weight_decay=0.0001 \
    --distortions=False \
    --summary_verbosity=0 \
    --use_datasets=False \
    --winograd_nonfused=True \
    --fp16_loss_scale=10.0 \
    --num_eval_epochs=1 \
    --eval_during_training_every_n_epochs=1 \
    --datasets_num_private_threads=5
