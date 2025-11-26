import math

import einops
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module
from transformers import BertConfig, BertLayer

import os
import mmaction
from mmaction.models import build_model
from mmcv import Config


class CSN(nn.Module):
    def __init__(self):
        super().__init__()
        mmaction_root = os.path.dirname(os.path.abspath(mmaction.__file__))
        config_file = os.path.join(mmaction_root, os.pardir, 'configs/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb.py')
        cfg = Config.fromfile(config_file)
        cfg.model.backbone.pretrained = None

        model = build_model(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        del model.cls_head
        self.model = model

    def forward(self, x):
        x = einops.rearrange(x, 'b t c h w -> b c t h w')
        """csn backbone need (B, C, T, H, W) format"""
        x = self.model.extract_feat(x)
        return x


def temporal_features(inputs, temporal_model, k, mode='left'):
    """(b c t)"""
    B = inputs.shape[0]
    L = inputs.shape[-1]

    padded_inputs = F.pad(inputs, pad=(0, math.ceil(L / k) * k - L), mode='replicate')
    # pad_L = padded_inputs.shape[-1]

    outputs = torch.zeros_like(padded_inputs)
    for offset in range(k):
        if mode == 'left':
            x = F.pad(padded_inputs, pad=(k - offset, 0), mode='replicate')[:, :, :-(k - offset)]
        elif mode == 'right':
            x = F.pad(padded_inputs, pad=(0, offset + 1), mode='replicate')[:, :, (offset + 1):]
        else:
            raise NotImplementedError

        # print(list(range(pad_L))[offset::k])
        seq = einops.rearrange(x, 'b c (k nw) -> (b nw) k c', k=k)
        h = temporal_model(seq)
        hidden_state = einops.rearrange(h, '(b nw) c -> b c nw', b=B)

        outputs[:, :, offset::k] = hidden_state

    outputs = einops.rearrange(outputs[:, :, :L], 'b c t -> b t c')  # (b t c)
    return outputs


class SelfAttention(Module):
    def __init__(self, dim):
        super().__init__()
        encoder_config = BertConfig(
            hidden_size=dim,
            num_attention_heads=8,
            intermediate_size=2048,
        )
        self.encoders = nn.ModuleList([BertLayer(encoder_config) for _ in range(6)])

    def forward(self, x):
        """b t c"""
        for encoder in self.encoders:
            x = encoder(x)[0]
        return x[:, 0]


class LeftRightFeatureExtractor(Module):
    def __init__(self, dim, stride, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.left_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=stride, bias=False)
        self.right_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=stride, bias=False)

    def forward(self, x):
        """x: (b c t)"""
        left_feats = self.left_conv(F.pad(x, pad=(self.kernel_size, 0), mode='replicate')[:, :, :-1])
        right_feats = self.right_conv(F.pad(x, pad=(0, self.kernel_size), mode='replicate')[:, :, 1:])

        feats = torch.cat([left_feats, right_feats], dim=1)  # (b c t)
        feats = einops.rearrange(feats, 'b c t -> b t c')  # (b t c)
        return feats


class E2EModel(Module):
    def __init__(self):
        super().__init__()
        in_feat_dim = 2048
        dim = 512
        self.kernel_size = 8
        self.backbone = CSN()
        self.trans_imgs_embedding = nn.Sequential(
            nn.Linear(in_feat_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        self.extractor = LeftRightFeatureExtractor(dim, stride=1, kernel_size=self.kernel_size)

        self.temporal_embedding = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
        )

        self.left_temporal_model = SelfAttention(dim)
        self.right_temporal_model = SelfAttention(dim)

        self.output = nn.Sequential(
            nn.Linear(dim * 2 + dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim * 2),
            nn.Dropout(0.2),
            nn.LayerNorm(dim * 2)
        )

        self.classifier = nn.Linear(
            in_features=dim * 2,
            out_features=1,
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.backbone(x)
        x = einops.rearrange(x, 'b c t h w -> (b t) c h w')

        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.trans_imgs_embedding(x)
        x = einops.rearrange(x, '(b t) c -> b c t', b=B)  # (4, 512, 100)
        origin_x = x

        x = self.temporal_embedding(x)

        feats = self.extractor(x)  # b t c

        left_temporal_feats = temporal_features(origin_x, self.left_temporal_model, k=self.kernel_size, mode='left')
        right_temporal_feats = temporal_features(origin_x, self.right_temporal_model, k=self.kernel_size, mode='right')
        #
        feats = torch.cat([feats, left_temporal_feats, right_temporal_feats], dim=-1)
        feats = self.output(feats)
        logits = self.classifier(feats)  # b t 1
        scores = torch.sigmoid(logits)[:, :, 0]
        return scores
