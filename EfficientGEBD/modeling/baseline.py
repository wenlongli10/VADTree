import itertools
import math

import einops
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm
from torchvision import models
from torchvision.ops.misc import FrozenBatchNorm2d
from utils.distribute import is_main_process
from utils.eval import early_exit_by_threshold
from .diff_former import DiffFormer, DiffMixer
from .resnet import ResNetXX
from .backbone import CSN, VideoMAEv2, TSM ,CSNR50
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


def pairwise_minus_l2_distance(out, n=0, group=1):
    if n > 0:
        out = torch.diff(out, n=n, dim=2)
        out = torch.nn.functional.pad(out, (0, 0, 1, 0), mode='replicate')
    c = out.shape[-1]
    # [b, nl*g, nf, C//g]
    group_out = torch.cat(torch.split(out, c // group, dim=-1), dim=1)
    x, y = group_out, group_out
    # x, y: (bs, num_layers, num_frames, num_channels)
    x = x.unsqueeze(3).detach()
    # ([4, 3, 100, 256]) -> ([4, 3, 1, 100, 256])
    y = y.unsqueeze(2)
    l2_dist = torch.sqrt(torch.sum((x - y) ** 2, dim=-1) + 1e-8)
    # (bs, num_layers, num_frames, feature_length)
    l2_dist = nn.InstanceNorm2d(l2_dist.size(1))(l2_dist)
    return -l2_dist


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
    def __init__(self, dim, num_layers, num_blocks, num_windows, group=1, resnet_type=1, is_basic=False, similarity_func='cosine', offset=0):
        super(DiffHead, self).__init__()
        self.dim = dim
        self.out_channels = dim * 1
        self.group = group
        self.similarity_func = similarity_func
        self.offset = offset
        self.is_basic = is_basic

        self.nl = num_layers
        self.nb = num_blocks
        self.nw = num_windows

        self.t_conv = nn.ModuleList([nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1, groups=dim), nn.ReLU(inplace=True)) for _ in range(3)])

        # self.diff_mixer = DiffMixer(dim)
        # self.ddm_encoder = ResNetXX(self.nl * group * num_blocks, dim, resnet_type=resnet_type)

        # self.diff_former = DiffFormer(dim=dim, diff_idx=num_layers-1, num_blocks=num_blocks)

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
        # self.out_conv = nn.Sequential(
        #     nn.Conv1d(dim * self.nl, dim, kernel_size=1, groups=dim),
        #     nn.BatchNorm1d(dim),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(dim, dim, 1)
        # )

        if not self.is_basic:
            self.crossse1_1d = CrossSE_1d(dim)
            self.crossse2_1d = CrossSE_1d(dim)
            self.fuse = nn.Sequential(
                nn.Linear(dim * 2, dim, bias=True), nn.ReLU(inplace=True),
            )
            # self.fuse = nn.Sequential(
            #     nn.Linear(dim, dim, bias=True), nn.ReLU(inplace=True),
            # )

    def forward(self, x_fps):

        out_list = []
        for i, xi in enumerate(x_fps):
            t_feat = 0.5 * (self.t_conv[i](xi) + xi)
            out_list.extend([t_feat])
        out_list = torch.stack(out_list, dim=1)  # (b*nw, nl, c, nf)

        if not self.is_basic:
            x = einops.rearrange(out_list, "(b nw) nl c nf -> (b nw nl) c nf", nw=self.nw, nl=self.nl)
            x = self.diff_mixer(x)
            diff_1d = x
            for i in range(self.nb):
                diff_1d = self.diff_encoder[i](diff_1d)
            x = einops.rearrange(x, "(b nw nl) c nf -> (b nw) nl c nf", nw=self.nw, nl=self.nl)
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

        # x = einops.rearrange(out_list, "(b nw) nl c nf -> (b nl nw) c nf", nw=self.nw, nl=self.nl)
        # out, out_diff_lists = self.diff_former(x)
        # diffs = []
        # nf = out_diff_lists[0].shape[-1]
        # for diff in out_diff_lists:
        #     nf_ = diff.shape[-1]
        #     if nf_ < nf:
        #         diff = torch.nn.functional.interpolate(diff, scale_factor=nf/nf_, mode='area')
        #     diff = einops.rearrange(diff, "(b nl nw) c nf -> (b nw) nl c nf", nw=self.nw, nl=self.nl)
        #     diffs.append(diff)
        # # (b*nw, nl*3, nf, c)
        # diffs = torch.cat(diffs, dim=1).permute(0, 1, 3, 2).contiguous()
        # x = diffs.permute(0, 2, 1, 3).contiguous()  # (b*nw, nf, nl*3, c)
        # similarity_func = self.similarity_func
        # if similarity_func == 'cosine':
        #     sim = F.cosine_similarity(x.unsqueeze(2), x.unsqueeze(1), dim=-1)  # (b*nw, nf, nf, nl*3)
        # else:
        #     raise NotImplemented
        # diff_map = sim.permute(0, 3, 1, 2).contiguous()  # (b*nw, nl*3, nf, nf)

        # diff_1d = einops.rearrange(diff_1d, "(bnw nl) c nf -> bnw (c nl) nf", nl=self.nl)
        # h = diff_1d.mean(-1)
        # diff_map = self.ddm_encoder(diff_map).mean(-1).mean(-1)  # (b*nw, 512)
        # h = einops.rearrange(diff_map, "(b nw) c -> b c nw", nw=self.nw)
        
        if not self.is_basic:
            diff_map = self.ddm_encoder(diff_map).mean(-1)
            diff_1d = einops.rearrange(diff_1d, "(bnw nl) c nf -> bnw nl c nf", nl=self.nl).mean(1)
            img2ddm = self.crossse1_1d(diff_1d, diff_map).mean(-1)
            ddm2img = self.crossse2_1d(diff_map, diff_1d).mean(-1)
            # (b*nw, 1024)
            h = torch.cat([img2ddm, ddm2img], dim=1)
            # (b, 512, nw)
            h = self.fuse(h).reshape(-1, self.dim, self.nw)
        else:
            diff_map = self.ddm_encoder(diff_map).mean(-1).mean(-1)  # (b*nw, 512)
            h = einops.rearrange(diff_map, "(b nw) c -> b c nw", nw=self.nw)
        return h


class BaseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone_name = cfg.MODEL.BACKBONE.NAME
        self.dataset = cfg.DATASETS.TRAIN[0].split('_')[0]
        self.fpn_start_idx = cfg.MODEL.FPN_START_IDX
        self.cat_prev = cfg.MODEL.CAT_PREV
        assert self.fpn_start_idx < (max(cfg.MODEL.HEAD_CHOICE)+1)
        if self.backbone_name == 'csn':
            self.backbone = CSN().backbone
            in_feat_dim = 2048
            self.fpn_layers = [self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4][:max(cfg.MODEL.HEAD_CHOICE)+1]
        elif self.backbone_name=='csn_r50':
            self.backbone = CSNR50().backbone
            in_feat_dim = 2048
            self.fpn_layers = [self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4][:max(cfg.MODEL.HEAD_CHOICE)+1]
        elif 'resnet' in self.backbone_name:
            self.backbone = getattr(models, self.backbone_name)(pretrained=True, norm_layer=FrozenBatchNorm2d)
            in_feat_dim = self.backbone.fc.in_features
            for param in itertools.chain(self.backbone.conv1.parameters(), self.backbone.bn1.parameters()):
                param.requires_grad = False
            del self.backbone.fc
            self.fpn_layers = [self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4][:max(cfg.MODEL.HEAD_CHOICE)+1]
            self.layer_dim = [64, 128, 256, 512] if self.backbone_name in ['resnet18','resnet34'] else None
        elif 'tsm' in self.backbone_name:
            self.backbone = TSM(T=cfg.INPUT.SEQUENCE_LENGTH).backbone
            in_feat_dim = self.backbone.fc.in_features
            for param in itertools.chain(self.backbone.conv1.parameters(), self.backbone.bn1.parameters()):
                param.requires_grad = False
            del self.backbone.fc
            self.fpn_layers = [self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4][:max(cfg.MODEL.HEAD_CHOICE)+1]
        elif self.backbone_name == 'mae':
            self.backbone = VideoMAEv2().backbone
            in_feat_dim = 384
            self.fpn_layers = None
            self.mae_feat_out = nn.Sequential(
                nn.Conv1d(in_feat_dim, cfg.MODEL.DIMENSION, 1), nn.ReLU(inplace=True))
            assert len(cfg.MODEL.HEAD_CHOICE) == 1, 'only 1 head in mae!'
        assert len(cfg.MODEL.HEAD_CHOICE) == len(cfg.MODEL.LOSS_WEIGHT), 'num losses must match num heads!'

        self.dim = cfg.MODEL.DIMENSION
        self.k = cfg.MODEL.K   # half size, true size 2k+1
        self.window_size = 2 * self.k + 1
        self.lenghth = cfg.INPUT.SEQUENCE_LENGTH
        self.num_tslice = cfg.MODEL.NUM_TSLICE    #1
        self.head_choice = cfg.MODEL.HEAD_CHOICE
        self.num_heads = len(cfg.MODEL.HEAD_CHOICE)
        self.num_blocks = cfg.MODEL.NUM_BLOCKS
        self.resnet_type = cfg.MODEL.RESNET_TYPE
        self.is_basic = cfg.MODEL.IS_BASIC
        # self.num_windows = math.ceil(self.lenghth / self.k)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.x1_out = nn.Sequential(
            nn.Conv1d((in_feat_dim // 8) if self.backbone_name not in ['resnet18','resnet34'] else self.layer_dim[0], self.dim, 1), nn.ReLU(inplace=True))
        self.x2_out = nn.Identity() if self.backbone_name not in ['resnet18','resnet34'] else nn.Sequential(
            nn.Conv1d(self.layer_dim[1], self.dim, 1), nn.ReLU(inplace=True))
        # self.x2_out = nn.Sequential(
        #     nn.Conv1d(in_feat_dim // 4, self.dim, 1), nn.ReLU(inplace=True))
        self.x3_out = nn.Sequential(
            nn.Conv1d((in_feat_dim // 2) if self.backbone_name not in ['resnet18','resnet34'] else self.layer_dim[2], self.dim, 1), nn.ReLU(inplace=True))
        self.x4_out = nn.Sequential(
            nn.Conv1d(in_feat_dim if self.backbone_name not in ['resnet18','resnet34'] else self.layer_dim[3], self.dim, 1), nn.ReLU(inplace=True))
        self.x_out = [self.x1_out, self.x2_out, self.x3_out, self.x4_out]
        # self.num_layers = 3 * 3
        self.diff_head = nn.ModuleList([DiffHead(dim=self.dim, num_layers=(i+1-self.fpn_start_idx if self.cat_prev else 1), 
            num_blocks=self.num_blocks, num_windows=self.lenghth//self.num_tslice, resnet_type=self.resnet_type, is_basic=self.is_basic) for i in cfg.MODEL.HEAD_CHOICE])
        self.classifiers = nn.ModuleList([nn.Sequential(
            nn.Conv1d(self.dim, self.dim, 3, 1, 1),
            nn.PReLU(),
            nn.Conv1d(self.dim, self.dim, 3, 1, 1),
            nn.PReLU(),
            nn.Conv1d(self.dim, 1, 1)) \
            for _ in range(self.num_heads)])


    def forward(self, inputs, targets=None):
        """
        Args:
            inputs(dict): imgs (B, T, C, H, W), frame_masks (B, T);
            targets: (B, T);
        Returns:
        """
        #***********************************************
        #While testing flops, change it into: imgs=inputs, else:imgs=inputs[imgs]
        imgs = inputs['imgs']
        # imgs=inputs
        #***********************************************

        B = imgs.shape[0]
        T = imgs.shape[1]

        if 'resnet' in self.backbone_name or 'tsm' in self.backbone_name:
            imgs = einops.rearrange(imgs, 'b t c h w -> (b t) c h w')
            x = self.backbone.conv1(imgs)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            # x = self.backbone.layer1(x)
        elif 'csn' in self.backbone_name:
            imgs = einops.rearrange(imgs, 'b t c h w -> b c t h w')
            x = self.backbone.conv1(imgs)
            x = self.backbone.maxpool(x)
            # x = self.backbone.layer1(x)
        elif self.backbone_name == 'mae':
            imgs = einops.rearrange(imgs, 'b t c h w -> b c t h w')
            xs = []
            for t_slice in range(0, T, T//self.num_tslice):
                x = imgs[:, :, t_slice:t_slice+T//self.num_tslice]
                x = self.backbone(x)
                x = self.avg_pool(x).squeeze(-2,-1)
                x = self.mae_feat_out(x)
                xs.append(x)  # (b, c, ts)
            x_fps = torch.cat(xs, dim=-1)  # (b, c, t)

        if self.fpn_layers is not None:
            xs = []   # feature pyramid
            for i, layer in enumerate(self.fpn_layers):
                x = layer(x)
                if 'resnet' in self.backbone_name or 'tsm' in self.backbone_name:
                    xi = x
                else:
                    xi = einops.rearrange(x, 'b c t h w -> (b t) c h w')

                xi = self.avg_pool(xi).squeeze(-2,-1)
                xi = einops.rearrange(xi, "(b t) c -> b c t", b=B)
                xi = self.x_out[i](xi)
                #x1=[2,512,100]   x2=[2,128,100]
                xs.append(xi)

            x_fp = torch.cat(xs, dim=1)
            x_fps = einops.rearrange(x_fp, "b (nl c) t -> (b nl) c t", b=B, c=self.dim)

        # slice to local windows
        x_fps = F.pad(x_fps, pad=(self.k, self.k), mode='replicate').unsqueeze(2)  # (b*nl, c, t+2k)
        x_fps = F.unfold(x_fps, kernel_size=(1, self.window_size))
        x_fps = einops.rearrange(x_fps, "(b nl) (c nf) t -> nl b t c nf", b=B, c=self.dim)

        logits_list = []
        for head, diff_head, clf in zip(self.head_choice, self.diff_head, self.classifiers):
            feats = []
            for t_slice in range(0, T, T//self.num_tslice):
                x_fpts = x_fps[:, :, t_slice:t_slice+T//self.num_tslice]
                x_fpts = einops.rearrange(x_fpts, "nl b ts c nf -> nl (b ts) c nf")
                feats.append(diff_head(x_fpts[self.fpn_start_idx:head+1] if self.cat_prev else x_fpts[head:head+1]))  # (b, c, ts)
            feats = torch.cat(feats, dim=-1)  # (b, c, t)
            logits = clf(feats)
            logits_list.append(logits)  # (b 1 t)

        
        if self.training:
            losses = []
            targets = targets.to(logits_list[0].dtype)
            if self.dataset != 'TAPOS':
                targets = prepare_gaussian_targets(targets)
            targets = targets.view(-1)

            for logits in logits_list:
                logits = logits.view(-1)
                if 'frame_masks' in inputs:
                    masks = inputs['frame_masks'].view(-1)
                    logits = logits[masks]
                    targets_valid = targets[masks]
                else:
                    targets_valid = targets
                loss = F.binary_cross_entropy_with_logits(logits, targets_valid)
                losses.append(loss)

            return losses

        scores = []
        for logits in logits_list:
            scores.append(torch.sigmoid(logits).flatten(1))
        # (b, 3, t)
        return torch.stack(scores, dim=1)
