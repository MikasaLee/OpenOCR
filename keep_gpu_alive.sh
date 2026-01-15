#!/bin/bash

cd /lirunrui/OpenOCR || exit 1

PYTHON=/lirunrui/miniconda/envs/openocr/bin/python
TORCHRUN=/lirunrui/miniconda/envs/openocr/bin/torchrun

while true; do
    echo "[$(date)] start eval" >> gpu_keepalive.log

    CUDA_VISIBLE_DEVICES=0,1 $TORCHRUN --nproc_per_node=2 --master-port=25071 /lirunrui/OpenOCR/tools/train_rec.py --c /lirunrui/OpenOCR/configs_bnu_en/rec/crnn/crnn_ctc_test.yml >> gpu_keepalive.log

    echo "[$(date)] finish eval, sleep 4h" >> gpu_keepalive.log
    sleep 14400
done
