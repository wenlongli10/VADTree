torchrun  EffSoccerNet/main.py \
--SoccerNet_path EffSoccerNet/data/R50_L4_5fps \
--features  R50_L4_5fps.npy \
--expname r50l4_gatherchunksamples \
--test_only \
--resume EffSoccerNet/r50l4_gatherchunksamples_/model_best.pth.tar