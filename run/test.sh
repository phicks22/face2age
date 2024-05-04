#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ~/anaconda3/envs/pytorch

loss=("mse" "huber" "log_cosh" "hubosh")

cd /mnt/scratch/f0106094/face2age
for fn in "${loss[@]}"; do
    python -u test.py \
    -batch_size 1 \
    -weights results/pretrained_weights/model__loss-"fn"__epochs-20__batch-64.pytorch \
    --test_set \
    -outfile "fn"__test-set.csv
done

