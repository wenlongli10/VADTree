import sys, os, json

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
# os.environ["CUDA_VISIBLE_DEVICES"] = "8,9"

import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy, math

import warnings
from decord import VideoReader, cpu
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from VLM_utils import args_to_dict
import psutil
from VLM_prompt import *

import torch.multiprocessing as mp

# mp.set_start_method('spawn', force=True)
import cv2
import argparse

# os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:128"
warnings.filterwarnings("ignore")
# 查看每个 GPU 的名称
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")




def parse_args():
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument("--pretrained", type=str,
        default = "/root/autodl-tmp/model_hub/LLaVA-Video-7B-Qwen2",
        help="Path to the pretrained model."
    )
    parser.add_argument("--video_root", type=str,
            # default=None,
            default = "/root/autodl-fs/lwl/data/UCF_Crime_test/test_videos",
            help="Path to the folder containing videos for inference. If None, will be set according to json_path."
            )
    parser.add_argument("--json_path", type=str,
        default = '../result/UCF_Crime_test/EGEBD_x2x3x4_r50_eff_split_out_th0.5_peak_dfs_kmeans_1_0.4'
                  '/dfs_coarse_sences'
                  '.json',
        help="Path to the JSON file containing HGTree results (coarse or fine)."
    )
    parser.add_argument("--dataset_clip", type=list,
        default = None,
        # default = [0,99],
        # default = [100,199],
        # default = [200,289],
        help="dataset clip for infer"
    )
    parser.add_argument("--prompt_flag", type=str,
        default = "prior_q",
        # default = "q1",
        # default = "q2",
        # default = "q3",
        help="prompt flag for different questions."
    )
    args = parser.parse_args()

    # 根据 json_path 自动设置不同数据集的配置
    if 'XD' in args.json_path:
        # if args.video_root is None: args.video_root = "/root/autodl-fs/lwl/data/XD_test/test_videos/"
        if args.prompt_flag == 'prior_q':
            args.prompt_flag = 'xd_prior_q'
    elif 'UCF' in args.json_path:
        # if args.video_root is None: args.video_root = "/root/autodl-fs/lwl/data/UCF_Crime_test/test_videos"
        if args.prompt_flag == 'prior_q':
            args.prompt_flag = 'ucf_prior_q'
    elif 'MSAD' in args.json_path:
        # if args.video_root is None: args.video_root = "/root/autodl-fs/lwl/data/MSAD_test/test_videos/"
        if args.prompt_flag == 'prior_q':
            args.prompt_flag = 'msad_prior_q'
    else:
        raise ValueError('Invalid JSON file path')
    return args

def load_clip_video(vr, clip, max_frames_num, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 384, 384, 3))
    clip[1] = min(clip[1], len(vr))  # clip[1] is exclusive
    total_frame_num = int(max(clip[1]-clip[0], 1))
    video_time = total_frame_num / vr.get_avg_fps()
    fps = vr.get_avg_fps()
    frame_idx = [i for i in range(0, total_frame_num, 1)]
    frame_time = [i/fps for i in frame_idx]
    frame_idx = [i + int(clip[0]) for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_time = np.array(frame_time)[uniform_sampled_frames].tolist()
        uniform_sampled_frames = uniform_sampled_frames + int(clip[0])
        frame_idx = uniform_sampled_frames.tolist()
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time

# 自定义数据集类
class VideoClipDataset(Dataset):
    def __init__(self, vid_json_dict, old_data_dict, video_root, maxf, preprocess, question,pad=0,
        force_sample=True):
        self.vid_json_dict = vid_json_dict
        self.old_data_dict = old_data_dict
        self.video_root = video_root
        self.maxf = maxf
        self.preprocess = preprocess
        self.question = question
        self.pad = pad
        self.force_sample = force_sample

        self.vid_json_dict_list = []
        # _vid_json_dict_list = list(self.vid_json_dict.items())
        self.cup_num = os.cpu_count()
        self.cur_cpu = 0
        self.scenes_limit = 200
        self.scenes_start = 0

        self._data_len = 0
        self.vid_idx = 0

        for k, v in self.vid_json_dict.items():
            if self.old_data_dict!=None: # 有旧数据
                if k in self.old_data_dict['vid_captions'].keys() and len(vid_json_dict[k]['scenes'])==len(
                    self.old_data_dict['vid_captions'][k]): # 旧数据中处理过这个视频，且
                    if list(self.old_data_dict['vid_captions'][k].keys())==[f'{i[0]}, {i[1]}' for i in vid_json_dict[
                        k]['scenes']] : # 场景完全相同
                        print(f'skip {k}') #说明旧数据完成了，无需再次处理
                        continue
            self.vid_json_dict_list.append((k,v))
            self._data_len += math.ceil(len(vid_json_dict[k]['scenes'])/self.scenes_limit)
        print(f'{len(self.vid_json_dict_list)} videos need to process, data len is {self._data_len}.')

    def __len__(self):
        return self._data_len

    def __getitem__(self, _):
        (k, v) = self.vid_json_dict_list[self.vid_idx]

        vid_path = os.path.join(self.video_root, k)

        pid = os.getpid()
        # 获取当前进程的 CPU 亲和性（绑定的 CPU 核心）
        process = psutil.Process(pid)
        # 获取当前进程正在运行的 CPU 核心
        current_cpu = process.cpu_num()
        vr = VideoReader(vid_path, ctx=cpu(current_cpu), num_threads=1)

        out = []
        while True:
            vid_seg = v['scenes'][self.scenes_start]
            # video_path = "XXXX"
            # vr = VideoReader(vid_path, ctx=cpu(0),num_threads=1)
            # assert vr.get_avg_fps() == 24.
            vid_seg[0] = max(0, vid_seg[0]- self.pad)
            video,frame_time,video_time = load_clip_video(vr,clip=vid_seg, max_frames_num=self.maxf,
                force_sample=self.force_sample)
            # video = video[:,40:,:200]

            video = self.preprocess(video, return_tensors="pt")["pixel_values"].half()

            conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
            time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video)} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
            question_ = DEFAULT_IMAGE_TOKEN + f"\n{time_instruciton}\n{self.question}."
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question_)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)

            out.append((self.vid_idx, video, input_ids, k, str(vid_seg)[1:-1]))


            self.scenes_start +=1

            # 判断是否需要继续读取当前视频的下一个场景，还是切换到下一个视频
            if self.scenes_start == len(v['scenes']): # 如果下一个scenes索引等于当前视频场景数，那么说明这个视频读取完成了，更新视频索引，重置场景索引
                self.vid_idx +=1
                self.scenes_start = 0
                return out
            elif len(out)>=self.scenes_limit: # 视频读取未完成且超过scenes的限制个数了，则不重置所有索引，先传输数据，防止内存OOM
                print(f'\t vid {k} The limit on the number of scenes that can be loaded at a single time for a single '
                      f'video is '
                      f'triggered ({self.scenes_limit}) !')
                return out
    def mask_vid(self,vid_name):
        masked_vid = []
        for idx, vid_ in enumerate(self.vid_json_dict_list):
            if vid_name == vid_[0]:
                masked_vid.append(vid_)


if __name__ == "__main__":
    args = parse_args()

    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    # device_map = "balanced_low_0"
    tokenizer, model, image_processor, max_length = load_pretrained_model(args.pretrained, None, model_name,
        torch_dtype="float16",
        device_map=device_map)  # Add any other thing you want to pass in llava_model_args
    model.eval()

    json_path = args.json_path
    video_root = args.video_root

    question = eval(args.prompt_flag)

    with open(json_path, 'r') as file:
        json_data = json.load(file)

    pad = 0
    # output_dir = '/root/autodl-tmp/data/XD_test/autoshot_103out_th0.3/LLaVA-Video-7B-Qwen2_/'
    # output_dir = '/root/autodl-tmp/data/diy/LLaVA-Video-7B-Qwen2/'
    # output_dir = '/root/autodl-tmp/data/UCF_Crime_test/autoshot_out_th0.3/LLaVA-Video-7B-Qwen2/'
    output_dir = os.path.join(os.path.dirname(json_path), f'{os.path.basename(args.pretrained)}'
                                                          f'_{args.prompt_flag}')
    if 'coarse' in json_path:
        output_dir += '_coarse'
    if 'fine' in json_path:
        output_dir += '_fine'
    if pad !=0:
        output_dir += f'_pad{pad}'

    os.makedirs(output_dir, exist_ok=True)



    maxf = 64 # default max_num_frames=64   双卡
    max_new_tokens = 512 # default max_new_tokens = 4096
    batch_size = 1
    num_workers = 1  # 根据 CPU 核心数调整
    force_sample = False

    # with open('/root/autodl-tmp/data/UCF_Crime_test/EGEBD_basic_r50_out_th0.5/LLaVA-Video-7B-Qwen2/maxf64_out_Please describe in one sentence what happened in the video..json', 'r', encoding='utf-8') as file:
    #     old_data = json.load(file)
    # out_json_path = os.path.join(output_dir, f"maxf{maxf}_out_prior_question.json")
    out_json_path = os.path.join(output_dir, f"maxf{maxf}_{args.prompt_flag}_{question[:10]}.json")

    if args.dataset_clip!=None:
        [s, e] = args.dataset_clip
        json_data = dict(list(json_data.items())[s:e+1])
    if len(json_data.keys()) not in [800, 290, 240]:
        out_json_path = out_json_path.replace('.json', f'_{s}_{e}.json')
    print('-----save to :', out_json_path)


    infer_out = {}
    old_data = None
    infer_out['vid_captions'] = {}
    if os.path.exists(out_json_path): # 继续之前的结果进行处理
        with open(out_json_path, 'rb') as f:  #
            # 必须用二进制模式打开[1,6](@ref)
            old_data = json.load(f)
        infer_out['vid_captions'] = old_data['vid_captions']

    dataset = VideoClipDataset(json_data, old_data, video_root, maxf, image_processor.preprocess, question,
         pad=pad, force_sample=force_sample)


    infer_out['config'] = {
        'args': args_to_dict(args,max_depth=5),
        'question':question,
        'model_path': args.pretrained,
        'maxf': maxf,
        'json_path': json_path,
        'pad': pad,
        'force_sample': force_sample,
    }

    data_loader = DataLoader(
        dataset,
        # batch_sampler=,
        # sampler=[idx,],
        batch_size=batch_size,
        shuffle=False,  # 是否打乱数据
        # num_workers=num_workers,  # 使用多进程加载数据
        num_workers=num_workers,  # 使用多进程加载数据,当前仅支持后台的单进程传输，因为涉及到batch间的数据切片
        pin_memory=True  # 如果使用 GPU，建议启用 pin_memory 加速数据传输
    )

    vid_captions = {}
    for outs in data_loader:
        # vid_out = {}

        k = outs[0][3][0]
        if k not in infer_out['vid_captions'].keys():
            infer_out['vid_captions'][k] = {}
        if k not in vid_captions.keys():
            vid_captions[k] = {}
        # input_ids_list = []

        with torch.no_grad():
            for i, out in enumerate(outs):
                (idx, video,input_ids,k, vid_seg_str) = out
                idx, video,input_ids,k, vid_seg_str = (int(idx), video[0], input_ids[0], k[0],
                vid_seg_str[0])
                # torch.cuda.empty_cache()
                cont = model.generate(
                    input_ids.to(device),
                    images=[video.to(device)],
                    modalities= ["video"],
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=max_new_tokens,
                )
                text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
                # print(text_outputs)
                vid_captions[k][vid_seg_str] = text_outputs

                # print(f'{idx} {k} {i}', f'{datetime.now().strftime("%H:%M:%S")}', f'Input video: {video.shape} '
                #                                                                f'frames: {vid_seg_str} '
                #                                                                f'input_ids:',input_ids[0].shape, '\n \t', text_outputs)
                print(f'{idx} {datetime.now().strftime("%H:%M:%S")} {list(json_data.keys()).index(k)} {k} {i} {vid_seg_str}')

        if len(vid_captions[k].keys()) == len(json_data[k]['scenes']): # 当前视频完全处理了才存进去
            infer_out['vid_captions'][k] = vid_captions[k]
            # # 每个视频的结果汇组成的json进行保存
            with open(out_json_path, 'w', encoding='utf-8') as json_out:
                # 序列化数据到文件，设置 ensure_ascii=False 以支持非ASCII字符
                # indent 参数用于格式化输出，使 JSON 文件更易读
                json.dump(infer_out, json_out, ensure_ascii=False, indent=4)
    print('-----save to :', os.path.abspath(out_json_path))
