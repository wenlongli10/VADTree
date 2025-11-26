# ******************testing EfficientGEBD-ResNet50-L2L3L4******************
torchrun --nproc_per_node 8 \
--master_port 1111 train.py \
--expname test \
--test-only \
--resume output/Kinetics-GEBD/x2x3x4_r50/model_best.pth \
MODEL.BACKBONE.NAME 'resnet50' \
MODEL.CAT_PREV True \
MODEL.FPN_START_IDX 1 \
MODEL.HEAD_CHOICE [3] \
MODEL.IS_BASIC False
#**************************************************************************