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
from .diff_former import DiffFormer
from .resnet import ResNet18, RepVGG8
from .backbone import CSN, VideoMAEv2
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=8):
        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # (T, C)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, T, C)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return x


class PositionalEncodingLearned(nn.Module):
    def __init__(self, dim, size=8):
        super(PositionalEncodingLearned, self).__init__()
        self.embed = nn.Embedding(size, dim)

    def forward(self, x):
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


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


class DiffHead(nn.Module):
    def __init__(self, dim, num_layers, num_blocks, num_windows, group=1, similarity_func='cosine', offset=0):
        super(DiffHead, self).__init__()
        self.dim = dim
        self.out_channels = dim * 1
        self.group = group
        self.similarity_func = similarity_func
        self.offset = offset

        self.nl = num_layers
        self.nb = num_blocks
        self.nw = num_windows

        # self.short_conv = nn.ModuleList([nn.Sequential(
        #     nn.Conv1d(dim, dim, 1), nn.ReLU(inplace=True)) for _ in range(3)])
        # self.middle_conv = nn.ModuleList([nn.Sequential(
        #     nn.Conv1d(dim, dim, 3, padding=1, groups=dim), nn.ReLU(inplace=True)) for _ in range(3)])
        # self.long_conv = nn.ModuleList([nn.Sequential(
        #     nn.Conv1d(dim, dim, 3, padding=1, groups=dim), nn.ReLU(inplace=True)) for _ in range(3)])
        self.t_conv = nn.ModuleList([nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1, groups=dim), nn.ReLU(inplace=True)) for _ in range(3)])

        self.diff_former = DiffFormer(dim=dim, diff_idx=num_layers-1, num_blocks=num_blocks)
        self.ddm_encoder = ResNet18(self.nl * group * num_blocks, dim)
        self.out_conv = nn.Sequential(
            nn.Conv1d(dim * self.nl, dim, kernel_size=1, groups=dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, 1)
        )
        self.crossse1_1d = CrossSE_1d(dim)
        self.crossse2_1d = CrossSE_1d(dim)
        self.fuse = nn.Sequential(
            nn.Linear(dim * 2, dim, bias=True), nn.ReLU(inplace=True),
        )

    # (nl, b*t, c, nf)  nw=num_windows, equals t
    def forward(self, x_fps):

        out_list = []
        for i, xi in enumerate(x_fps):
            # short_feat = 0.5 * (self.short_conv[i](xi) + xi)
            # middle_feat = 0.5 * (self.middle_conv[i](xi) + xi)
            # long_feat = 0.5 * (self.long_conv[i](middle_feat) + middle_feat)
            # out_list.extend([short_feat, middle_feat, long_feat])

            t_feat = 0.5 * (self.t_conv[i](xi) + xi)
            out_list.extend([t_feat])

        out_list = torch.stack(out_list, dim=1)  # (b*nw, nl, c, nf)

        x = einops.rearrange(out_list, "(b nw) nl c nf -> (b nl nw) c nf", nw=self.nw, nl=self.nl)

        out, out_diff_lists = self.diff_former(x)

        diffs = []
        nf = out_diff_lists[0].shape[-1]
        for diff in out_diff_lists:
            nf_ = diff.shape[-1]
            if nf_ < nf:
                diff = torch.nn.functional.interpolate(diff, scale_factor=nf/nf_, mode='area')
            diff = einops.rearrange(diff, "(b nl nw) c nf -> (b nw) nl c nf", nw=self.nw, nl=self.nl)
            diffs.append(diff)
        # (b*nw, nl*3, nf, c)
        diffs = torch.cat(diffs, dim=1).permute(0, 1, 3, 2).contiguous()

        x = diffs.permute(0, 2, 1, 3).contiguous()  # (b*nw, nf, nl*3, c)
        
        similarity_func = self.similarity_func
        if similarity_func == 'cosine':
            sim = F.cosine_similarity(x.unsqueeze(2), x.unsqueeze(1), dim=-1)  # (b*nw, nf, nf, nl*3)
        else:
            raise NotImplemented
        diff_map = sim.permute(0, 3, 1, 2).contiguous()  # (b*nw, nl*3, nf, nf)

        # torch.save(diff_map.cpu(), f'diff_map_{self.nl//3-1}.pt')

        diff_map = self.ddm_encoder(diff_map)

        # (b*nw, 512, nf_down_down)
        diff_map = torch.mean(diff_map, dim=-1)

        out = einops.rearrange(out, "(b nl) c nf -> b (c nl) nf", nl=self.nl)
        # (b*nw, 512, 3)
        out = self.out_conv(out)

        img2ddm = self.crossse1_1d(out, diff_map).mean(-1)
        ddm2img = self.crossse2_1d(diff_map, out).mean(-1)

        # (b*nw, 1024)
        h = torch.cat([img2ddm, ddm2img], dim=1)
        # (b, 512, nw)
        h = self.fuse(h).reshape(-1, self.dim, self.nw)
        return h


def SPoS(inputs, ban, k, nl):
    """(b c t)"""
    B = inputs.shape[0] // (nl // 3)
    L = inputs.shape[-1]

    padded_inputs = F.pad(inputs, pad=(0, math.ceil(L / k) * k - L), mode='replicate')
    pad_L = padded_inputs.shape[-1]

    # outputs = torch.zeros_like(padded_inputs)
    outputs = torch.zeros(B, ban.out_channels, pad_L, dtype=inputs.dtype, device=inputs.device)
    for offset in range(k):
        left_x = F.pad(padded_inputs, pad=(k - offset, 0), mode='replicate')[:, :, :-(k - offset)]
        right_x = F.pad(padded_inputs, pad=(0, offset + 1), mode='replicate')[:, :, (offset + 1):]
        left_seq = einops.rearrange(left_x, 'b c (nw k) -> (b nw) k c', k=k)
        right_seq = einops.rearrange(right_x, 'b c (nw k) -> (b nw) k c', k=k)
        mid_seq = einops.rearrange(padded_inputs[:, :, offset::k], 'b c nw -> (b nw) 1 c')

        h = ban(left_seq, mid_seq, right_seq)  # (b nw) c
        hidden_state = einops.rearrange(h, '(b nw) c -> b c nw', b=B)

        outputs[:, :, offset::k] = hidden_state

    outputs = outputs[:, :, :L]  # (b c t)
    return outputs


def pad_exit_frames(x_reorg, remain_idx, B, T=100):
    remain_idx = np.array(remain_idx)
    for b in range(B):
        # pad the beginning frame
        if b*T not in remain_idx:
            idx = remain_idx[remain_idx > b*T]
            if any(idx):
                nearest_right_idx = idx[0]
                x_reorg[b*T] = x_reorg[nearest_right_idx]
                remain_idx = np.append(remain_idx, b*T)
        # pad the ending frame
        if b*T+T-1 not in remain_idx:
            idx = remain_idx[remain_idx < b*T+T-1]
            if any(idx):
                nearest_left_idx = idx[-1]
                x_reorg[b*T] = x_reorg[nearest_left_idx]
                remain_idx = np.append(remain_idx, b*T+T-1)
    remain_idx = np.sort(remain_idx)

    diff = remain_idx[1:] - remain_idx[:-1]
    diff_length = diff[diff > 1].tolist()
    diff_pos = [remain_idx[i] for i in np.where(diff > 1)[0]]

    for pos, len in zip(diff_pos, diff_length):
        # x_pad = (x_reorg[pos] + x_reorg[pos + len]) / 2
        # x_reorg[pos + 1: pos + len] = x_pad[None].repeat(len - 1, 1, 1, 1)
        x_reorg[pos + 1: pos + len // 2] = x_reorg[pos][None].repeat(len // 2 - 1, 1, 1, 1)
        x_reorg[pos + len // 2: pos + len] = x_reorg[pos + len][None].repeat(len - len // 2, 1, 1, 1)

    return x_reorg


class E2EModelDiff(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone_name = cfg.MODEL.BACKBONE.NAME
        self.dataset = cfg.DATASETS.TRAIN[0].split('_')[0]
        if self.backbone_name == 'csn':
            self.backbone = CSN().backbone
            in_feat_dim = 2048
            self.fpn_layers = [self.backbone.layer2, self.backbone.layer3, self.backbone.layer4][:max(cfg.MODEL.HEAD_CHOICE)+1]
        elif 'resnet' in self.backbone_name:
            self.backbone = getattr(models, self.backbone_name)(pretrained=True, norm_layer=FrozenBatchNorm2d)
            in_feat_dim = self.backbone.fc.in_features
            for param in itertools.chain(self.backbone.conv1.parameters(), self.backbone.bn1.parameters()):
                param.requires_grad = False
            del self.backbone.fc
            self.fpn_layers = [self.backbone.layer2, self.backbone.layer3, self.backbone.layer4][:max(cfg.MODEL.HEAD_CHOICE)+1]
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
        self.num_tslice = cfg.MODEL.NUM_TSLICE
        self.head_choice = cfg.MODEL.HEAD_CHOICE
        self.num_heads = len(cfg.MODEL.HEAD_CHOICE)
        self.num_blocks = cfg.MODEL.NUM_BLOCKS
        # self.num_windows = math.ceil(self.lenghth / self.k)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.x1_out = nn.Sequential(
        #     nn.Conv1d(in_feat_dim // 8, self.dim, 1), nn.ReLU(inplace=True))
        self.x2_out = nn.Identity()
        # self.x2_out = nn.Sequential(
        #     nn.Conv1d(in_feat_dim // 4, self.dim, 1), nn.ReLU(inplace=True))
        self.x3_out = nn.Sequential(
            nn.Conv1d(in_feat_dim // 2, self.dim, 1), nn.ReLU(inplace=True))
        self.x4_out = nn.Sequential(
            nn.Conv1d(in_feat_dim, self.dim, 1), nn.ReLU(inplace=True))
        self.x_out = [self.x2_out, self.x3_out, self.x4_out]
        # self.num_layers = 3 * 3
        self.diff_head = nn.ModuleList([DiffHead(dim=self.dim, num_layers=1, num_blocks=self.num_blocks, \
            num_windows=self.lenghth//self.num_tslice) for i in cfg.MODEL.HEAD_CHOICE])
        self.classifiers = nn.ModuleList([nn.Sequential(
            nn.Conv1d(self.dim, self.dim, 3, 1, 1),
            nn.PReLU(),
            nn.Conv1d(self.dim, self.dim, 3, 1, 1),
            nn.PReLU(),
            nn.Conv1d(self.dim, 1, 1)) \
            for _ in range(self.num_heads)])
        
        self.th = cfg.TEST.THRESHOLD
        self.thh = cfg.TEST.THH
        self.thl = cfg.TEST.THL
        self.ignore = cfg.TEST.IGNORE
        self.pad_ignore = cfg.TEST.PAD_IGNORE
        self.sigma = cfg.SOLVER.SIGMA
    
    
    def forward(self, inputs, targets=None):
        """
        Args:
            inputs(dict): imgs (B, T, C, H, W), frame_masks (B, T);
            targets: (B, T);
        Returns:
        """

        # torch.cuda.synchronize()
        # time0 = time.time()

        imgs = inputs['imgs']

        B = imgs.shape[0]
        T = imgs.shape[1]

        if 'resnet' in self.backbone_name:
            imgs = einops.rearrange(imgs, 'b t c h w -> (b t) c h w')
            x = self.backbone.conv1(imgs)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
        elif self.backbone_name == 'csn':
            imgs = einops.rearrange(imgs, 'b t c h w -> b c t h w')
            x = self.backbone.conv1(imgs)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
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

        # torch.cuda.synchronize()
        # time1 = time.time()
        # print(f'stem time: {(time1-time0)*100}ms')
        # last_time = time1

        if self.fpn_layers is not None:
            xs = []   # feature pyramid
            for i, layer in enumerate(self.fpn_layers):
                x = layer(x)
                if 'resnet' in self.backbone_name:
                    xi = x
                else:
                    xi = einops.rearrange(x, 'b c t h w -> (b t) c h w')

                # torch.cuda.synchronize()
                # print(f'layer {i+2} time: {(time.time()-last_time)*1000}ms')
                # last_time = time.time()

                xi = self.avg_pool(xi).squeeze(-2,-1)
                xi = einops.rearrange(xi, "(b t) c -> b c t", b=B)
                xi = self.x_out[i](xi)
                xs.append(xi)

            # torch.cuda.synchronize()
            # print(f'x{i+2}_out time: {(time.time()-last_time)*1000}ms')
            # last_time = time.time()

            x_fp = torch.cat(xs, dim=1)
            x_fps = einops.rearrange(x_fp, "b (nl c) t -> (b nl) c t", b=B, c=self.dim)

        # slice to local windows
        x_fps = F.pad(x_fps, pad=(self.k, self.k), mode='replicate').unsqueeze(2)  # (b*nl, c, t+2k)
        x_fps = F.unfold(x_fps, kernel_size=(1, self.window_size))
        x_fps = einops.rearrange(x_fps, "(b nl) (c nf) t -> nl b t c nf", b=B, c=self.dim)

        # torch.cuda.synchronize()
        # print(f'slice_all time: {(time.time()-last_time)*1000}ms')
        # last_time = time.time()

        logits_list = []
        for head, diff_head, clf in zip(self.head_choice, self.diff_head, self.classifiers):
            feats = []
            for t_slice in range(0, T, T//self.num_tslice):
                x_fpts = x_fps[:, :, t_slice:t_slice+T//self.num_tslice]
                x_fpts = einops.rearrange(x_fpts, "nl b ts c nf -> nl (b ts) c nf")
                feats.append(diff_head(x_fpts[head:head+1]))  # (b, c, ts)***************************************
            feats = torch.cat(feats, dim=-1)  # (b, c, t)
            logits = clf(feats)
            logits_list.append(logits)  # (b 1 t)
        

        # assert 1==2

        #     torch.cuda.synchronize()
        #     print(f'head {i+2} time: {(time.time()-last_time)*1000}ms')
        #     last_time = time.time()
        
        # assert 1==2
        
        if self.training:
            losses = []
            targets = targets.to(logits_list[0].dtype)
            if self.dataset != 'TAPOS':
                targets = prepare_gaussian_targets(targets)
            # hard_targets_from_th = (targets > self.th).long()
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

            # return losses, logits_list, hard_targets_from_th
            return losses

        scores = []
        for logits in logits_list:
            scores.append(torch.sigmoid(logits).flatten(1))
        # (b, 3, t)
        return torch.stack(scores, dim=1)


    def dynamic_inference(self, inputs):
        """
        Args:
            inputs(dict): imgs (B, T, C, H, W), frame_masks (B, T);
            targets: (B, T);
        Returns:
        """
        assert not self.training, 'dynamic network only work during model inference!'

        # torch.cuda.synchronize()
        # time0 = time.time()
        
        imgs = inputs['imgs']

        B = imgs.shape[0]
        T = imgs.shape[1]
        
        imgs = einops.rearrange(imgs, 'b t c h w -> (b t) c h w')

        x = self.backbone.conv1(imgs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)

        x_fp = torch.tensor([]).cuda()   # feature pyramid

        remain_idx_last = []
        psu_remain_idx_last = []
        boundarys = [[] for _ in range(B)]
        exit_percentages = []

        # torch.cuda.synchronize()
        # time1 = time.time()
        # print(f'stem time: {(time1-time0)*100}ms')
        # last_time = time1

        for i in range(self.num_heads):
            x = self.fpn_layers[i](x)

            # torch.cuda.synchronize()
            # print(f'layer {i+2} time: {(time.time()-last_time)*1000}ms')
            # last_time = time.time()

            # padding exited frames
            if i > 0:
                x_reorg = torch.zeros([B*T] + list(x.shape[1:])).cuda()
                # flatten list of lists
                remain_idx = list(chain(*remain_idx))
                x_reorg[remain_idx] = x
                if remain_idx:
                    x_reorg = pad_exit_frames(x_reorg, remain_idx, B, T)

                # torch.cuda.synchronize()
                # print(f'layer {i+2} pad time: {(time.time()-last_time)*1000}ms')
                # last_time = time.time()

            else:
                x_reorg = x

            xi = self.avg_pool(x_reorg).squeeze()

            xi = einops.rearrange(xi, "(b t) c -> b c t", b=B)
            xi = self.x_out[i](xi)

            # torch.cuda.synchronize()
            # print(f'x{i+2}_out time: {(time.time()-last_time)*1000}ms')
            # last_time = time.time()

            x_fp = torch.cat([x_fp, xi], dim=1)
            x_fps = einops.rearrange(x_fp, "b (nl c) t -> (b nl) c t", b=B, c=self.dim)

            # torch.cuda.synchronize()
            # print(f'x{i+2}_rearrange time: {(time.time()-last_time)*1000}ms')
            # last_time = time.time()

            # slice to local windows
            x_fps = F.pad(x_fps, pad=(self.k, self.k), mode='replicate').unsqueeze(2)  # (b*nl, c, t+2k)
            x_fps = F.unfold(x_fps, kernel_size=(1, self.window_size))
            x_fps = einops.rearrange(x_fps, "(b nl) (c nf) t -> nl (b t) c nf", b=B, c=self.dim)

            # torch.cuda.synchronize()
            # print(f'x{i+2}_slice time: {(time.time()-last_time)*1000}ms')
            # last_time = time.time()
            
            feats = self.diff_head[i](x_fps)  # (b, c, t)
            logits = self.classifiers[i](feats)  # (b, 1, t)
            pred = torch.sigmoid(logits).flatten(1).cpu()  # (b, t)
            # torch.cuda.synchronize()
            # print(f'head {i+2} time: {(time.time()-last_time)*1000}ms')
            # last_time = time.time()

            # early exit with fixed threshold
            x_remain = []
            remain_idx = []
            psu_remain_idx = []   # remove negative predictions introduced by padding
            exit_percentage = []
            x_reorg = einops.rearrange(x_reorg, "(b t) c h w -> b t c h w", b=B)
            for b in range(B):
                # NOTE: the idx is not aligned with real video
                boundary_idx_per_vid, exit_idx_per_vid, psu_exit_idx_per_vid = early_exit_by_threshold(
                    seq_scores=pred[b], thh=self.thh[i], thl=self.thl[i], ignore=self.ignore, pad_ignore=self.pad_ignore)
                remain_idx_per_vid = list(set([*range(T)]) - set(list(chain(*exit_idx_per_vid))))
                psu_remain_idx_per_vid = list(set([*range(T)]) - set(list(chain(*psu_exit_idx_per_vid))))
                if remain_idx_last:
                    remain_idx_per_vid = list(set(remain_idx_per_vid) & set([r - T * b for r in remain_idx_last[b]]))
                    boundary_idx_per_vid = list(set(boundary_idx_per_vid) & set([r - T * b for r in remain_idx_last[b]])
                                                & set(psu_remain_idx_last[b]))
                boundarys[b].extend(boundary_idx_per_vid)
                # map idxs to sequence
                remain_idx.append([ridx + T * b for ridx in remain_idx_per_vid])
                psu_remain_idx.append(psu_remain_idx_per_vid)
                x_vid_remain = x_reorg[b][remain_idx_per_vid]
                x_remain.append(x_vid_remain)
                exit_percentage.append(1. - x_vid_remain.shape[0] / T)
            remain_idx_last = remain_idx
            psu_remain_idx_last = psu_remain_idx
            x = torch.cat(x_remain, dim=0)  # (\sum_b{t_b}, c, h, w)
            
            if i < self.num_heads - 1:
                exit_percentages.append(exit_percentage)
            
        #     torch.cuda.synchronize()
        #     print(f'head {i+2} exit time: {(time.time()-last_time)*1000}ms')
        #     last_time = time.time()
        
        # assert 1==2
        
        boundarys = [sorted(b) for b in boundarys]
        exit_percentages = np.array(exit_percentages).T

        # print(boundarys)
        # assert 1==2

        return boundarys, exit_percentages
