import itertools
import math

import einops
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm
from torchvision import models
from torchvision.ops.misc import FrozenBatchNorm2d
from diff_former import DiffFormer, DiffMixer
from resnet import ResNetXX
from itertools import chain
import numpy as np
import time


def prepare_gaussian_targets(targets, sigma=1):
    gaussian_targets = []
    for batch_idx in range(targets.shape[0]):
        t = targets[batch_idx]
        axis = torch.arange(len(t), device=targets.device)
        gaussian_t = torch.zeros_like(t)
        indices, = torch.nonzero(t, as_tuple=True)
        for i in indices:
            g = torch.exp(-(axis - i) ** 2 / (2 * sigma * sigma))
            gaussian_t += g

        gaussian_t = gaussian_t.clamp(0, 1)
        # gaussian_t /= gaussian_t.max()
        gaussian_targets.append(gaussian_t)
    gaussian_targets = torch.stack(gaussian_targets, dim=0)
    return gaussian_targets


class CrossSE_1d(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CrossSE_1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        b, c, _ = x.size()
        x = self.avg_pool(x).view(b, c)
        x = self.fc(x).view(b, c, 1)
        return y * x.expand_as(y)


class DiffMixer(nn.Module):

    def __init__(
        self,
        dim,
        diff_idx=0,
        kernel_size=1,
    ):
        super().__init__()
        self.dim = dim
        self.diff_idx = diff_idx
        self.kernel_size = kernel_size

        self.pre_norm = nn.BatchNorm1d(dim)
        self.d = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(dim, dim, kernel_size, groups=1),
                    nn.BatchNorm1d(dim),
                    nn.GELU()
                )
                for _ in range(2)
            ]
        )

    def diff(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.diff(x, n=1, dim=2)
        x = torch.nn.functional.pad(x, (1, 0), mode='replicate')
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_identity = x
        x = self.pre_norm(x)
        x_d0 = self.d[0](x)
        x_d1 = self.d[1](self.diff(x))
        x_d = x_d0 + x_d1
        x = x_identity + x_d
        return x


class DiffHead(nn.Module):
    def __init__(self, dim, num_layers, num_blocks, group=1, resnet_type=1, is_basic=False, similarity_func='cosine', offset=0):
        super(DiffHead, self).__init__()
        self.dim = dim
        self.out_channels = dim * 1
        self.group = group
        self.similarity_func = similarity_func
        self.offset = offset
        self.is_basic = is_basic

        self.nl = num_layers
        self.nb = num_blocks

        self.t_conv = nn.ModuleList([nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1, groups=dim), nn.ReLU(inplace=True)) for _ in range(3)])

        self.diff_mixer = DiffMixer(dim)

        if not self.is_basic:
            self.diff_encoder = nn.ModuleList()
            for _ in range(self.nb):
                self.diff_encoder.append(
                    nn.Sequential(
                        nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
                        nn.Conv1d(dim, dim, kernel_size=1, groups=1),
                        nn.BatchNorm1d(dim),
                        nn.ReLU(inplace=True)
                    )
                )
        self.ddm_encoder = ResNetXX(self.nl * group * (1 if self.is_basic else 2), dim, resnet_type=resnet_type)

        if not self.is_basic:
            self.crossse1_1d = CrossSE_1d(dim)
            self.crossse2_1d = CrossSE_1d(dim)
            self.fuse = nn.Sequential(nn.Linear(dim * 2, dim, bias=True), nn.ReLU(inplace=True))


    def forward(self, x_fps, nw):

        out_list = []
        for i, xi in enumerate(x_fps):
            t_feat = 0.5 * (self.t_conv[i](xi) + xi)
            out_list.extend([t_feat])
        out_list = torch.stack(out_list, dim=1)  # (b*nw, nl, c, nf)

        if not self.is_basic:
            x = einops.rearrange(out_list, "(b nw) nl c nf -> (b nw nl) c nf", nw=nw, nl=self.nl)
            x = self.diff_mixer(x)
            diff_1d = x
            for i in range(self.nb):
                diff_1d = self.diff_encoder[i](diff_1d)
            x = einops.rearrange(x, "(b nw nl) c nf -> (b nw) nl c nf", nw=nw, nl=self.nl)
            diffs = torch.cat([out_list, x], dim=1)
        else:
            diffs = out_list

        diffs = diffs.permute(0, 1, 3, 2).contiguous()
        x = diffs.permute(0, 2, 1, 3).contiguous()  # (b*nw, nf, nl*2, c)
        similarity_func = self.similarity_func
        if similarity_func == 'cosine':
            sim = F.cosine_similarity(x.unsqueeze(2), x.unsqueeze(1), dim=-1)  # (b*nw, nf, nf, nl)
        else:
            raise NotImplemented
        diff_map = sim.permute(0, 3, 1, 2).contiguous()  # (b*nw, nl, nf, nf)
        
        if not self.is_basic:
            diff_map = self.ddm_encoder(diff_map).mean(-1)
            diff_1d = einops.rearrange(diff_1d, "(bnw nl) c nf -> bnw nl c nf", nl=self.nl).mean(1)
            img2ddm = self.crossse1_1d(diff_1d, diff_map).mean(-1)
            ddm2img = self.crossse2_1d(diff_map, diff_1d).mean(-1)
            # (b*nw, 1024)
            h = torch.cat([img2ddm, ddm2img], dim=1)
            # (b, 512, nw)
            h = self.fuse(h).reshape(-1, self.dim, nw)
        else:
            diff_map = self.ddm_encoder(diff_map).mean(-1).mean(-1)  # (b*nw, 512)
            h = einops.rearrange(diff_map, "(b nw) c -> b c nw", nw=nw)
        return h


class BaseModel(nn.Module):
    def __init__(self, args, cfg):
        super().__init__()
        in_feat_dim = 2048

        self.dim = cfg.MODEL.DIMENSION
        self.k = args.receptive_field * args.framerate   # half size, true size 2k+1
        self.window_size = 2 * self.k + 1
        # self.lenghth = args.chunk_size * args.framerate
        self.num_tslice = 1
        self.num_blocks = cfg.MODEL.NUM_BLOCKS
        self.resnet_type = cfg.MODEL.RESNET_TYPE
        self.is_basic = cfg.MODEL.IS_BASIC
        self.gaussian_sigma = cfg.MODEL.SIGMA

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.x4_out = nn.Sequential(
            nn.Conv1d(in_feat_dim, self.dim, 1), nn.ReLU(inplace=True))
        self.diff_head = DiffHead(dim=self.dim, num_layers=1, num_blocks=self.num_blocks,\
                                  resnet_type=self.resnet_type, is_basic=self.is_basic)
        self.classifiers = nn.Sequential(
            nn.Conv1d(self.dim, self.dim, 3, 1, 1),
            nn.PReLU(),
            nn.Conv1d(self.dim, self.dim, 3, 1, 1),
            nn.PReLU(),
            nn.Conv1d(self.dim, 1, 1))


    def forward(self, inputs, targets=None, mask=None):
        """
        Args:
            inputs(dict): imgs (B, T, C);
            targets: (B, T);
        Returns:
        """
        imgs = inputs

        B, T, C = imgs.shape
        
        x_fps = self.x4_out(imgs.permute(0, 2, 1))   # (b, c, t)

        # slice to local windows
        x_fps = F.pad(x_fps, pad=(self.k, self.k), mode='replicate').unsqueeze(2)  # (b, c, 1, t+2k)
        x_fps = F.unfold(x_fps, kernel_size=(1, self.window_size))
        x_fps = einops.rearrange(x_fps, "b (c nf) t -> b t c nf", c=self.dim)

        feats = []
        # self.num_tslice = 1 if self.training else 10
        for t_slice in range(0, T, T//self.num_tslice):
            x_fpts = x_fps[:, t_slice:t_slice+T//self.num_tslice]
            ts = x_fpts.shape[1]
            x_fpts = einops.rearrange(x_fpts, "b ts c nf -> (b ts) c nf").unsqueeze(0)
            feats.append(self.diff_head(x_fpts, nw=ts))   # (b, c, ts)
        feats = torch.cat(feats, dim=-1)   # (b, c, t)
        logits = self.classifiers(feats)   # (b, 1, t)

        if self.training:
            targets = targets.to(logits.dtype)
            # soft label for supervision
            targets = prepare_gaussian_targets(targets, self.gaussian_sigma)
            targets = targets.view(-1)
            logits = logits.view(-1)
            if mask is not None:
                mask = mask.view(-1)
                targets = targets[mask]
                logits = logits[mask]
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            return loss
        else:
            score = torch.sigmoid(logits).flatten(1)   # (b, t)
            return score
