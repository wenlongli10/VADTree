# ********************training EfficientGEBD-CSN-L1L2**********************
torchrun --nproc_per_node 8 \
--master_port 1111 train.py \
--expname x1x2_csn_eff \
MODEL.BACKBONE.NAME 'csn' \
MODEL.CAT_PREV True \
MODEL.FPN_START_IDX 0 \
MODEL.HEAD_CHOICE [1] \
MODEL.IS_BASIC False
#**************************************************************************