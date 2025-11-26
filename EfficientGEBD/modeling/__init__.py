import torch
from .config import _C as cfg

from .baseline import BaseModel
from .e2e_model_diff_former import E2EModelDiff


def build_model(cfg):
    if cfg.MODEL.NAME == 'BaseModel':
        model = BaseModel(cfg)
    elif cfg.MODEL.NAME == 'E2EModelDiff':
        model = E2EModelDiff(cfg)
    else:
        raise NotImplemented(f'No such model: {cfg.MODEL.NAME}')

    if cfg.MODEL.SYNC_BN:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    return model
