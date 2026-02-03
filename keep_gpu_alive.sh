#!/bin/bash

cd /lirunrui/OpenOCR || exit 1

PYTHON=/lirunrui/miniconda/envs/openocr/bin/python
TORCHRUN=/lirunrui/miniconda/envs/openocr/bin/torchrun

# 检查GPU上是否有运行的进程
check_gpu_process() {
    # 检查GPU上是否有计算进程
    if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q "[0-9]"; then
        return 0  # 有进程运行
    else
        return 1  # 没有进程运行
    fi
}

while true; do

    # 检查GPU上是否有模型在运行
    if ! check_gpu_process; then
        echo "[$(date)] GPU上没有进程运行，启动训练保活" >> gpu_keepalive.log
        
        CUDA_VISIBLE_DEVICES=0,1 $TORCHRUN --nproc_per_node=2 --master-port=25071 /lirunrui/OpenOCR/tools/train_rec.py --c /lirunrui/OpenOCR/configs_bnu_en/rec/crnn/crnn_ctc_test.yml >> gpu_keepalive.log
        
        echo "[$(date)] 训练完成" >> gpu_keepalive.log
    else
        echo "[$(date)] GPU上有进程在运行，无需启动保活任务" >> gpu_keepalive.log
    fi

    # 每隔5小时检查一次，防止6小时不用GPU导致服务器自动停机
    sleep 18000  # 5小时
done
