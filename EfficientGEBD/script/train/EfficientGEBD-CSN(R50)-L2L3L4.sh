# ******************training EfficientGEBD-CSN(R50)-L2L3L4*****************
torchrun --nproc_per_node 8 \
--master_port 1111 train.py \
--expname x2x3x4_csn_r50_eff \
MODEL.BACKBONE.NAME 'csn_r50' \
MODEL.CAT_PREV True \
MODEL.FPN_START_IDX 1 \
MODEL.HEAD_CHOICE [3] \
MODEL.IS_BASIC False \
SOLVER.LR 5e-3 \
SOLVER.MAX_EPOCHS 20
#**************************************************************************