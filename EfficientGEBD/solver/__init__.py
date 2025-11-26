import torch
from timm.scheduler import CosineLRScheduler
from torch.optim.lr_scheduler import MultiStepLR


def build_optimizer(cfg, params):
    kwargs = dict(lr=cfg.SOLVER.LR,
                  weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    if cfg.SOLVER.OPTIMIZER == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.MOMENTUM, **kwargs)
    elif cfg.SOLVER.OPTIMIZER == 'AdamW':
        optimizer = torch.optim.AdamW(params, **kwargs)
    elif cfg.SOLVER.OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(params, **kwargs)
    else:
        raise NotImplementedError
    return optimizer
