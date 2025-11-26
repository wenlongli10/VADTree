import os

import einops
import torch
import torch.nn as nn

import mmengine
from mmaction.registry import MODELS
from mmengine.registry import init_default_scope

import torchvision
from torchvision.ops.misc import FrozenBatchNorm2d

# from csn import csn152


class CSN(nn.Module):
    def __init__(self):
        super().__init__()
        # config_path = 'CSN-pretrained/CSN-configs/R152/ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb.py'
        config_path = 'CSN-pretrained/CSN-configs/R152/ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb.py'
        # config_path = '/workspace/AR/mmaction2/configs/recognition/csn/ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb.py'
        checkpoint_path = 'CSN-pretrained/CSN-ckpt/R152/ircsn_from_scratch_r152_ig65m_20200807-771c4135.pth'
        #checkpoint_path = '/workspace/AR/mmaction2/ckpt/ircsn_from_scratch_r152_ig65m_20200807-771c4135.pth'
        config = mmengine.Config.fromfile(config_path)
        init_default_scope(config.get('default_scope', 'mmaction'))
        # if hasattr(config.model, 'backbone') and config.model.backbone.get(
        #     'pretrained', None):
        #     config.model.backbone.pretrained = None
        # config.model.backbone.bn_frozen = True
        model = MODELS.build(config.model)
        ckpt = torch.load(checkpoint_path)
        # print('####################')
        # print([k for k in ckpt['state_dict'].keys()])
        # print('####################')
        
        # new_ckpt = {}
        # for k, v in ckpt.items():
        #     new_ckpt['backbone.' + k] = v
        
        model.backbone.load_state_dict(ckpt)

        # model.backbone.out_indices = [1, 2, 3]
        self.backbone = model.backbone

    def forward(self, x):
        # x = einops.rearrange(x, 'b t c h w -> b c t h w')
        """csn backbone need (B, C, T, H, W) format"""

        x = einops.rearrange(x, '(b t) c h w -> b c t h w', t=100)

        x = self.backbone.conv1(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        return x

class CSNR50(nn.Module):
    def __init__(self):
        super().__init__()
        config_path = 'CSN-pretrained/CSN-configs/R152/ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb.py'
        checkpoint_path = 'CSN-pretrained/CSN-ckpt/R152/ircsn_from_scratch_r152_ig65m_20200807-771c4135.pth'

        # config_path ='CSN-pretrained/CSN-configs/R50/ircsn_ig65m-pretrained-r50-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb.py'
        #config_path ='/workspace/AR/mmaction2/configs/recognition/csn/ircsn_ig65m-pretrained-r50-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb.py'
        # checkpoint_path ='CSN-pretrained/CSN-ckpt/R50/ircsn_from_scratch_r50_ig65m_20210617-ce545a37.pth'
        #checkpoint_path ='/workspace/AR/mmaction2/ckpt/ircsn_from_scratch_r50_ig65m_20210617-ce545a37.pth'
        config = mmengine.Config.fromfile(config_path)
        init_default_scope(config.get('default_scope', 'mmaction'))
        # if hasattr(config.model, 'backbone') and config.model.backbone.get(
        #     'pretrained', None):
        #     config.model.backbone.pretrained = None
        # config.model.backbone.bn_frozen = True
        model = MODELS.build(config.model)
        ckpt = torch.load(checkpoint_path)
        # print('####################')
        # print([k for k in ckpt['state_dict'].keys()])
        # print('####################')
        
        # new_ckpt = {}
        # for k, v in ckpt.items():
        #     new_ckpt['backbone.' + k] = v
        
        model.backbone.load_state_dict(ckpt)

        # model.backbone.out_indices = [1, 2, 3]
        self.backbone = model.backbone

    def forward(self, x):
        # x = einops.rearrange(x, 'b t c h w -> b c t h w')
        """csn backbone need (B, C, T, H, W) format"""

        x = einops.rearrange(x, '(b t) c h w -> b c t h w', t=100)

        x = self.backbone.conv1(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        return x

class TSM(nn.Module):
    def __init__(self, T):
        super().__init__()

        self.backbone = getattr(torchvision.models, 'resnet50')(pretrained=True, norm_layer=FrozenBatchNorm2d)
        from .temporal_shift import make_temporal_shift
        make_temporal_shift(self.backbone, T, n_div=8, place='blockres', temporal_pool=False)

    def forward(self, x):
        x = self.backbone(x)
        return x


class VideoMAEv2(nn.Module):
    def __init__(self):
        super().__init__()
        config_path = '/workspace/AR/mmaction2/configs/recognition/videomaev2/vit-small-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400.py'
        checkpoint_path = '/workspace/AR/mmaction2/ckpt/vit-small-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-25c748fd.pth'
        config = mmengine.Config.fromfile(config_path)
        init_default_scope(config.get('default_scope', 'mmaction'))
        if hasattr(config.model, 'backbone') and config.model.backbone.get(
            'pretrained', None):
            config.model.backbone.pretrained = None
        config.model.backbone.bn_frozen = True
        model = MODELS.build(config.model)
        ckpt = torch.load(checkpoint_path)
        new_ckpt = {}
        for k, v in ckpt.items():
            if 'backbone' in k and 'patch_embed' not in k:
                new_ckpt[k.split('backbone.')[1]] = v
        model.backbone.load_state_dict(new_ckpt, strict=False)

        self.backbone = model.backbone

    def forward(self, x):
        x = einops.rearrange(x, 'b t c h w -> b c t h w')
        """csn backbone need (B, C, T, H, W) format"""
        x = self.backbone(x)
        # print([i.shape for i in x])
        # x = einops.rearrange(x, 'b c t h w -> b t c h w')
        return x


# class TSN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         mmaction_root = os.path.dirname(os.path.abspath(mmaction.__file__))
#         config_file = os.path.join(mmaction_root, os.pardir, 'configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py')
#         cfg = Config.fromfile(config_file)

#         model = build_model(
#             cfg.model,
#             train_cfg=cfg.get('train_cfg'),
#             test_cfg=cfg.get('test_cfg'))
#         state_dict = torch.load(os.path.join(mmaction_root, os.pardir, 'tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'))
#         print('load from tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth', flush=True)
#         model.load_state_dict(state_dict['state_dict'])
#         del model.cls_head
#         self.model = model

#     def forward(self, x):
#         B = x.shape[0]
#         x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
#         x = self.model.extract_feat(x)
#         x = einops.rearrange(x, '(b t) c h w -> b t c h w', b=B)
#         return x

class TEST_MODEL(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model
    def forward(self, x):
        print(x.shape)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.max_pool(x)
        print(x.shape)
        x = self.backbone.layer1(x)
        print(x.shape)
        x = self.backbone.layer2(x)
        print(x.shape)
        return x

if __name__ == '__main__':
    # model = CSN().cuda()
    model = TEST_MODEL(csn152(num_classes=400)).cuda()
    x = torch.rand(1, 3, 100, 224, 224).cuda()
    # x = einops.rearrange(x, 'b t c h w -> b c t h w')

    # from thop import profile
    # from torchvision.models import resnet152
    # model = TEST_MODEL(resnet152(num_classes=1000))
    # x = torch.rand(1, 3, 224, 224)
    # macs_1, params = profile(model, inputs=(x, ))
    # print(macs_1 / 1e9)

    # from torch_flops import TorchFLOPsByFX
    # flops_counter = TorchFLOPsByFX(model)
    # # flops_counter.graph_model.graph.print_tabular()
    # flops_counter.propagate(x)
    # # flops_counter.print_result_table()
    # flops_1 = flops_counter.print_total_flops(show=False)
    # print(flops_1 / 1e9)

    # import torchanalyse
    # unit = torchanalyse.Unit(unit_flop='mFLOP')
    # system = torchanalyse.System(
    #     unit,
    #     frequency=940,
    #     flops=123,
    #     onchip_mem_bw=900,
    #     pe_min_density_support=0.0001,
    #     accelerator_type="structured",
    #     model_on_chip_mem_implications=False,
    #     on_chip_mem_size=32,
    # )
    # x = torch.rand(100, 3, 224, 224).cuda()
    # result_2 = torchanalyse.profiler(model, x, system, unit)
    # flops_2 = sum(result_2['Flops (mFLOP)'].values) / 1e3
    # print(f"torchanalyse: {flops_2:.0f} FLOPs")

    from ptflops import get_model_complexity_info
    # from torchvision.models import resnet152
    # model = TEST_MODEL(resnet152(num_classes=1000))
    flops, params = get_model_complexity_info(model, (3,100,224,224),as_strings=True,print_per_layer_stat=True)
    print(flops, params)

    # from fvcore.nn import FlopCountAnalysis, parameter_count_table
    # # from torchvision.models import resnet50
    # # model = TEST_MODEL(resnet50(num_classes=1000))
    # # x = torch.rand(1, 3, 224, 224)
    # flops = FlopCountAnalysis(model, x)
    # print("FLOPs: ", flops.total() / 1e9)


    # x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
    # x = model(x)
    # print(x.shape)
