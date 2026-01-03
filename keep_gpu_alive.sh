#!/bin/bash

cd /lirunrui/OpenOCR || exit 1

PYTHON=/lirunrui/miniconda/envs/openocr/bin/python
TORCHRUN=/lirunrui/miniconda/envs/openocr/bin/torchrun

while true; do
    echo "[$(date)] start eval" >> gpu_keepalive.log

    CUDA_VISIBLE_DEVICES=0 $TORCHRUN \
        --nproc_per_node=1 \
        --master-port=25079 \
        tools/eval_rec_VisualC3_ids_test.py \
        --c ./configs_new_visualC3_ids/rec/crnn/crnn_ctc.yml \
        >> gpu_keepalive.log 2>&1

    echo "[$(date)] finish eval, sleep 4h" >> gpu_keepalive.log
    sleep 14400
done
