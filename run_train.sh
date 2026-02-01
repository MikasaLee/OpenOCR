# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25073 tools/train_rec.py --c ./configs_bnu_en/rec/crnn/crnn_ctc.yml
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25075 tools/train_rec.py --c ./configs_bnu_en/rec/svtrv2/svtrv2_smtr_gtc_rctc.yml


CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25073 tools/train_rec.py --c configs_new_visualC3_ids/rec/nrtr/nrtr_testdecoder.yml 
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25073 tools/train_rec.py --c configs_new_visualC3_ids/rec/nrtr/nrtr_testdecoder20.yml