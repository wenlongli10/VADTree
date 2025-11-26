# ******************training BasicGEBD-ResNet34-L2*************************
torchrun --nproc_per_node 8 \
--master_port 1111 train.py \
--expname x2_r34_basic \
MODEL.BACKBONE.NAME 'resnet34' \
MODEL.CAT_PREV False \
MODEL.FPN_START_IDX 1 \
MODEL.HEAD_CHOICE [1] \
MODEL.IS_BASIC True
#**************************************************************************