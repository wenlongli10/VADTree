import os

import einops
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import mmengine
from mmaction.registry import MODELS
from mmengine.registry import init_default_scope

import numpy as np
from decord import VideoReader, gpu, cpu
import glob
import shutil
from tqdm import tqdm

from torchvision.models import resnet50


class CSN(nn.Module):
    def __init__(self):
        super().__init__()
        config_path = 'CSN-pretrained/CSN-configs/R152/ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb.py'
        checkpoint_path = 'CSN-pretrained/CSN-ckpt/R152/ircsn_from_scratch_r152_ig65m_20200807-771c4135.pth'
        config = mmengine.Config.fromfile(config_path)
        init_default_scope(config.get('default_scope', 'mmaction'))
        model = MODELS.build(config.model)
        ckpt = torch.load(checkpoint_path)
        model.backbone.load_state_dict(ckpt)
        self.backbone = model.backbone

    @torch.no_grad()
    def extract_features(self, x):
        # x = einops.rearrange(x, 'b t c h w -> b c t h w')
        """csn backbone need (B, C, T, H, W) format"""

        x = einops.rearrange(x, '(b t) c h w -> b c t h w', b=1)

        x = self.backbone.conv1(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = x.mean([-2, -1])
        x = einops.rearrange(x, 'b c t -> (b t) c', b=1)
        return x


preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def process_video(video_path, target_fps=5):
    vr = VideoReader(video_path, ctx=cpu(1))
    original_fps = vr.get_avg_fps()
    total_frames = len(vr)
    sample_interval = int(original_fps / target_fps)
    
    frames = []
    for i in range(0, total_frames, sample_interval):
        frame = vr[i].asnumpy()
        frame = preprocess(frame)
        frames.append(frame)
    # (T, 3, 224, 224)
    frames_tensor = torch.stack(frames)
    return frames_tensor


# corrupted videos
skip_game = ["spain_laliga/2016-2017/2017-04-26 - 20-30 Barcelona 7 - 1 Osasuna", 
             "spain_laliga/2016-2017/2017-04-08 - 21-45 Malaga 2 - 0 Barcelona"]

if __name__ == '__main__':

    # model = CSN().cuda()
    model = resnet50(pretrained=True).cuda()
    model.eval()

    # video_dirs = glob.glob('/data1/SoccerNet/videos/*/*/*')[448:]   # skip 447 spain_laliga/2016-2017/2017-04-26 - 20-30 Barcelona 7 - 1 Osasuna
    # video_dirs = glob.glob('/data1/SoccerNet/videos/*/*/*')[448+12:]   # skip spain_laliga/2016-2017/2017-04-08 - 21-45 Malaga 2 - 0 Barcelona
    video_dirs = glob.glob('/data1/SoccerNet/videos/*/*/*')

    max_len = 1024

    for video_dir in tqdm(video_dirs, total=len(video_dirs)):
        if skip_game[0] in video_dir or skip_game[1] in video_dir:
            continue
        feats_dir_l2 = video_dir.replace('/videos', '/R50_L2_5fps')
        feats_dir_l4 = video_dir.replace('/videos', '/R50_L4_5fps')
        if not os.path.exists(feats_dir_l2):
            os.makedirs(feats_dir_l2)
        if not os.path.exists(feats_dir_l4):
            os.makedirs(feats_dir_l4)
        # for half in ['1', '2']:
        for half in ['2']:
            video_path = os.path.join(video_dir, half + '_224p.mkv')
            frames_tensor = process_video(video_path)
            total_t = frames_tensor.shape[0]
            features_l2, features_l4 = [], []
            for t in range(0, total_t, max_len):
                if t + max_len <= total_t:
                    images = frames_tensor[t:t+max_len].cuda()
                else:
                    images = frames_tensor[t:].cuda()
                with torch.no_grad():
                    x = model.conv1(images)
                    x = model.bn1(x)
                    x = model.relu(x)
                    x = model.maxpool(x)
                    x = model.layer1(x)
                    x = model.layer2(x)
                    features_l2.append(x.clone().cpu())
                    x = model.layer3(x)
                    x = model.layer4(x)
                    features_l4.append(x.clone().cpu())
                # features.append(model.extract_features(images).cpu())
            # features = torch.cat(features, dim=0).numpy()
            features_l2 = torch.cat(features_l2, dim=0).numpy()
            features_l4 = torch.cat(features_l4, dim=0).numpy()
            # save_path = os.path.join(feats_dir, half + '_CSN_5fps.npy')
            np.save(os.path.join(feats_dir_l2, half + '_R50_L2_5fps.npy'), features_l2)
            np.save(os.path.join(feats_dir_l4, half + '_R50_L4_5fps.npy'), features_l4)
        # shutil.copy(os.path.join(video_dir, 'Labels-cameras.json'), os.path.join(feats_dir_l2, 'Labels-cameras.json'))
        # shutil.copy(os.path.join(video_dir, 'Labels-cameras.json'), os.path.join(feats_dir_l4, 'Labels-cameras.json'))

