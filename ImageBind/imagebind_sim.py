# lavad
import argparse
import sys, os, json, copy, glob
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gc  # 添加垃圾回收模块
from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import warnings
from decord import VideoReader, cpu
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))

def compute_chunked_similarity(emb_a, emb_b, chunk_size=200):
    """
    分块计算两组嵌入向量间的余弦相似度，用于处理大型数据集时节省内存

    参数:
        emb_a: 第一组嵌入向量
        emb_b: 第二组嵌入向量
        chunk_size: 每块处理的向量数量

    返回:
        numpy数组，包含所有相似度结果
    """
    arr = np.zeros((emb_a.shape[0], emb_b.shape[0]))
    for i in range(0, emb_a.shape[0], chunk_size):
        chunk_end = min(i + chunk_size, emb_a.shape[0])
        arr[i:chunk_end] = F.cosine_similarity(
            emb_a[i:chunk_end].unsqueeze(1),
            emb_b.unsqueeze(0),
            dim=-1
        ).numpy()
    return arr


class VideoClipDataset(Dataset):
    def __init__(self, vid_json_dict,video_root, old_data=None):
        self.vid_json_dict = vid_json_dict
        self.video_root = video_root
        self.old_data = old_data

        self.vid_json_dict_list = list(self.vid_json_dict.items())
        self.cup_num = os.cpu_count()
        self.cur_cpu = 0

    def __len__(self):
        return len(self.vid_json_dict_list)

    def __getitem__(self, idx):

        (k, v) = self.vid_json_dict_list[idx]
        if self.old_data!=None:
            if k in self.old_data:
                return (idx, k, k ,k)
        video_path = os.path.join(self.video_root, k)

        # out = []
        text_lists, vid_intervals = [], []
        for kk,vv in v.items():
            vid_intervals.append( (int(float(kk.split(', ')[0])), int(float(kk.split(', ')[1]))) )
            text_lists.append(vv)

        # Load data
        text_inputs = data.load_and_transform_text(text_lists, 'cpu')
        vid_inputs = data.load_and_transform_video_seg_data_cache(video_path, vid_intervals ,'cpu')
        # vid_inputs = [glob.glob("./*.cache.pt")]

        # out.append((idx, k, text_inputs, vid_inputs))

        return (idx, k, text_inputs, vid_inputs)
    def mask_vid(self,vid_name):
        masked_vid = []
        for idx, vid_ in enumerate(self.vid_json_dict_list):
            if vid_name == vid_[0]:
                masked_vid.append(vid_)


if __name__ == "__main__":
    # Instantiate model
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_summary_json', type=str, 
                        default='../result/UCF_Crime_test/EGEBD_x2x3x4_r50_eff_split_out_th0.5_peak_dfs_kmeans_1_0.4/LLaVA-Video-7B-Qwen2_ucf_prior_q_coarse/maxf64_ucf_prior_q_Here is a .json',
                         help='video summary json path')
    parser.add_argument('--video_root', type=str, 
                        # default=None,
                        default = "/root/autodl-fs/lwl/data/UCF_Crime_test/test_videos",
                         help='Video root path, If None, will be set according to video_summary_json')
    args = parser.parse_args()

    video_summary_json = args.video_summary_json
    video_root = args.video_root

    # 自动识别视频根目录
    if args.video_root is None:
        if 'UCF_Crime' in video_summary_json:
            video_root = "/root/autodl-fs/lwl/data/UCF_Crime_test/test_videos"
        elif 'XD' in video_summary_json:
            video_root = "/root/autodl-fs/lwl/data/XD_test/test_videos/"
        elif 'MSAD' in video_summary_json:
            video_root = "/root/autodl-fs/lwl/data/MSAD_test/test_videos/"
        else:
            raise NotImplementedError("Please manually set the video_root for the dataset.")


    print(f'video_summary_json: {video_summary_json}')
    with open(video_summary_json) as f:
        all_video_summary_dict = json.load(f)


    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)



    output_pkl_path = os.path.join(os.path.dirname(video_summary_json), f'sim_{os.path.basename(video_summary_json)[:-4]}pkl')
    print(f'\t\t save to: {os.path.abspath(output_pkl_path)}')
    if os.path.exists(output_pkl_path):
        print(f'Load old data.... {output_pkl_path}')
        with open(output_pkl_path, 'rb') as f:
            # 必须用二进制模式打开[1,6](@ref)
            output_json = pickle.load(f)
    else:
        output_json = copy.deepcopy(all_video_summary_dict)
        del output_json['vid_captions']
        output_json['vid_sim'] = {}
        output_json['vid_feat'] = {}
        output_json['text_feat'] = {}

    # with open('/root/autodl-tmp/data/UCF_Crime_test/EGEBD_basic_r50_split_out_th0.3/LLaVA-Video-7B-Qwen2/sim_maxf64_out_Please describe in one sentence what happened in the video.buckup.pkl', 'rb') as f:  #
    #     # 必须用二进制模式打开[1,6](@ref)
    #     backup_data = pickle.load(f)
    # output_json['vid_sim'] = backup_data['vid_sim']

    dataset = VideoClipDataset(all_video_summary_dict['vid_captions'], video_root, old_data=output_json[
        'vid_sim'].keys())
    dataset.start = None
    # dataset.start = 200
    data_loader = DataLoader(
        dataset,
        # batch_sampler=,
        # sampler=[i for i in range(81, len(dataset), 1)],
        batch_size=1,
        shuffle=False,  # 是否打乱数据
        # num_workers=num_workers,  # 使用多进程加载数据
        num_workers=1,  # 使用多进程加载数据
        pin_memory=False,  # 如果使用 GPU，建议启用 pin_memory 加速数据传输
        # pin_memory=False  # 如果使用 GPU，建议启用 pin_memory 加速数据传输
    )
    with torch.no_grad():

        # for idx, (k,v) in enumerate(all_video_summary_dict['vid_captions'].items()):
        for outs in data_loader:
            # if dataset.tart!=None:
            #     if idx < dataset.start:
            #         print(f'skip {idx,k}')
            #         continue
            (idx, k, text_inputs, vid_inputs) = outs
            if outs[1][0] in output_json['vid_sim'].keys():
                print(f'skip {idx[0], outs[1][0]}')
                continue
            (idx, k, text_inputs, vid_inputs) = (idx[0], k[0], text_inputs[0], vid_inputs[0])
            output_json['vid_sim'][k] = {}

            # 如果文本输入长度大于200，则分块处理文本数据
            if len(text_inputs) > 200:
                print(f'----------------{len(text_inputs)} segs------------')  # 输出文本输入的长度
                emb_texts = []
                # 将文本输入划分成每块50个，逐块计算文本特征
                for i in torch.split(text_inputs, 50, dim=0):
                    emb_text = model({ModalityType.TEXT: i.to(device)})[ModalityType.TEXT]
                    emb_texts.append(emb_text)
                emb_texts = torch.cat(emb_texts, dim=0)  # 合并所有文本特征
                del text_inputs  # 删除原始文本输入释放内存
            else:
                # 文本输入长度较小时直接计算文本特征
                emb_texts = model({ModalityType.TEXT: text_inputs.to(device)})[ModalityType.TEXT]

            emb_vids = []
            # 如果vid_inputs是一个path列表，则说明使用了缓存文件，依次读取缓存文件并按照原方式处理信息
            if isinstance(vid_inputs, list) and all(isinstance(p[0], str) for p in vid_inputs):
                for path in vid_inputs:
                    # 分块处理视频输入，每块4个，逐块计算视频特征
                    temp_inputs = torch.load(path[0])
                    split_temp_inputs = torch.split(temp_inputs, 4, dim=0)
                    for i in split_temp_inputs:
                        emb_vid = model({ModalityType.VISION: i.to(torch.float32).to(device)})[ModalityType.VISION]
                        emb_vids.append(emb_vid)
                    del temp_inputs  # 删除临时变量
                    del split_temp_inputs  # 删除临时变量
                    torch.cuda.empty_cache()
                    gc.collect()  # 强制垃圾回收
            else:
                vid_inputs = torch.split(vid_inputs, 4, dim=0)
                for i in vid_inputs:
                    emb_vid = model({ModalityType.VISION: i.to(torch.float32).to(device)})[ModalityType.VISION]
                    emb_vids.append(emb_vid)
            emb_vids = torch.cat(emb_vids, dim=0)  # 合并所有视频特征



            if len(emb_vids)>1000:  # 大数据时先转CPU分块计算省内存
                print(f'----------------{len(emb_vids)} clips, sim分块计算------------')  # 输出视频输入的长度
                emb_vids, emb_texts = emb_vids.cpu(), emb_texts.cpu()
                # arr = F.cosine_similarity(emb_vids.unsqueeze(1),emb_texts.unsqueeze(0), dim=-1).numpy()
                # 将视频嵌入分成更小的块进行计算
                chunk_size = 200  # 可以根据实际内存情况调整

                arr = compute_chunked_similarity(emb_vids, emb_texts, chunk_size=200)
                # output_json['vid_sim'][k]['VxT'] = [[round(num, 2) for num in row] for row in arr]
                output_json['vid_sim'][k]['VxT'] = np.round(arr,3)

                arr = compute_chunked_similarity(emb_texts, emb_vids, chunk_size=200)
                output_json['vid_sim'][k]['TxV'] = np.round(arr,3)
                arr = compute_chunked_similarity(emb_vids, emb_vids, chunk_size=200)
                output_json['vid_sim'][k]['VxV'] = np.round(arr,3)
                arr = compute_chunked_similarity(emb_texts, emb_texts, chunk_size=200)
                output_json['vid_sim'][k]['TxT'] = np.round(arr,3)

                output_json['vid_feat'][k] = emb_vids.cpu().numpy()
                output_json['text_feat'][k] = emb_texts.cpu().numpy()
            else: # 小数据时边计算边转CPU
                # emb_vids, emb_texts = emb_vids.cpu(), emb_texts.cpu()
                arr = F.cosine_similarity(emb_vids.unsqueeze(1),emb_texts.unsqueeze(0), dim=-1).cpu().numpy()
                # output_json['vid_sim'][k]['VxT'] = [[round(num, 2) for num in row] for row in arr]
                output_json['vid_sim'][k]['VxT'] = np.round(arr,3)
                arr = F.cosine_similarity(emb_texts.unsqueeze(1),emb_vids.unsqueeze(0), dim=-1).cpu().numpy()
                output_json['vid_sim'][k]['TxV'] = np.round(arr,3)
                arr = F.cosine_similarity(emb_vids.unsqueeze(1),emb_vids.unsqueeze(0), dim=-1).cpu().numpy()
                output_json['vid_sim'][k]['VxV'] = np.round(arr,3)
                arr = F.cosine_similarity(emb_texts.unsqueeze(1),emb_texts.unsqueeze(0), dim=-1).cpu().numpy()
                output_json['vid_sim'][k]['TxT'] = np.round(arr,3)

                output_json['vid_feat'][k] = emb_vids.cpu().numpy()
                output_json['text_feat'][k] = emb_texts.cpu().numpy()
            print(f'{idx} {datetime.now().strftime("%H:%M:%S")} {k} {len(arr)} clips ')

            with open(f"{output_pkl_path}", "wb") as f:
                pickle.dump(output_json, f)

    print(f'video_summary_json: {os.path.abspath(video_summary_json)}')
    print(f'sim pkl save to: {os.path.abspath(output_pkl_path)}')
    # 清空.cache.pt后缀的缓存文件
    cache_files = glob.glob("./*.cache.pt")
    for cache_file in cache_files:
        os.remove(cache_file)

