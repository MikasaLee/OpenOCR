# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25073 tools/train_rec.py --c ./configs_bnu_en/rec/crnn/crnn_ctc.yml
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25075 tools/train_rec.py --c ./configs_bnu_en/rec/svtrv2/svtrv2_smtr_gtc_rctc.yml


# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25073 tools/train_rec.py --c configs_new_visualC3_ids/rec/nrtr/nrtr_testdecoder.yml 
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25073 tools/train_rec.py --c configs_new_visualC3_ids/rec/nrtr/nrtr_testdecoder20.yml

# VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25073 tools/train_rec.py --c configs_new_visualC3_textline/rec/aster/resnet31_lstm_aster_tps_on.yml
# VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25073 tools/train_rec.py --c configs_new_visualC3_textline/rec/aster/resnet31_lstm_aster_tps_on.yml
# VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25073 tools/train_rec.py --c configs_new_visualC3_textline/rec/aster/resnet31_lstm_aster_tps_on.yml
# VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25073 tools/train_rec.py --c configs_new_visualC3_textline/rec/aster/resnet31_lstm_aster_tps_on.yml
# VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25073 tools/train_rec.py --c configs_new_visualC3_textline/rec/aster/resnet31_lstm_aster_tps_on.yml
# VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25073 tools/train_rec.py --c configs_new_visualC3_textline/rec/aster/resnet31_lstm_aster_tps_on.yml


# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25073 tools/train_rec.py --c configs_new_visualC3_textline/rec/abinet/resnet45_trans_abinet_lang.yml
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25073 tools/train_rec.py --c configs_new_visualC3_textline/rec/parseq/vit_parseq.yml
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25073 tools/train_rec.py --c configs_new_visualC3_textline/rec/lister/focalsvtr_lister_wo_fem_maxratio12.yml
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25073 tools/train_rec.py --c configs_new_visualC3_textline/rec/smtr/focalsvtr_smtr.yml
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25073 tools/train_rec.py --c configs_new_visualC3_textline/rec/cppd/svtr_base_cppd_ch.yml
#CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25073 tools/train_rec.py --c configs_bctr/rec/svtrs/svtr_base_ctc.yml
#CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25073 tools/train_rec.py --c configs_bctr/rec/parseq/vit_parseq.yml
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25073 tools/train_rec.py --c configs_bctr/rec/smtr/focalsvtr_smtr.yml


# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25071 tools/train_rec.py --c configs_bnu_en/rec/svtrv2/svtrv2_smtr_gtc_rctc_padding.yml
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25071 tools/train_rec.py --c configs_bnu_ch/rec/svtrv2/svtrv2_smtr_gtc_rctc_padding.yml

# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25071 tools/train_rec.py --c configs_bnu_en/rec/eduocr/eduocr_padding.yml
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25071 tools/train_rec.py --c configs_bnu_ch/rec/eduocr/eduocr_padding.yml

# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25071 tools/train_rec.py --c configs_bnu_en/rec/eduocr/eduocr_2.yml
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25071 tools/train_rec.py --c configs_bnu_ch/rec/eduocr/eduocr_2.yml


# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25071 tools/train_rec.py --c configs_bnu_zh/rec/svtrv2/svtrv2_smtr_gtc_rctc.yml 
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25071 tools/train_rec.py --c configs_bnu_zh/rec/eduocr/eduocr.yml

# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25071 tools/train_rec.py --c configs_bnu_en/rec/eduocr/eduocr_2_rctc.yml
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25071 tools/train_rec.py --c configs_bnu_zh/rec/eduocr/eduocr_rctc.yml

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25071 tools/train_rec.py --c configs_bnu_en/rec/eduocr/eduocr_2_rctc_padding.yml
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=25071 tools/train_rec.py --c configs_bnu_zh/rec/eduocr/eduocr_rctc_padding.yml