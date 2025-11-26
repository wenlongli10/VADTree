import argparse
import json

from collections import defaultdict
import torch.distributed as dist
from modeling import cfg, build_model
from utils.distribute import is_main_process
import torch.backends.cudnn as cudnn
import random
from torchvision import transforms

from autoshot_utils import *
import os
from datetime import datetime
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 查看当前 GPU 的编号
# print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))

def make_inputs(inputs, device):
    keys = ['imgs', 'video_path', 'frame_masks']
    results = {}
    if isinstance(inputs, dict):
        for key in keys:
            if key in inputs:
                val = inputs[key]
                if isinstance(val, torch.Tensor):
                    val = val.to(device)
                results[key] = val
    elif isinstance(inputs, list):
        targets = defaultdict(list)
        for item in inputs:
            for key in keys:
                if key in item:
                    val = item[key]
                    targets[key].append(val)

        for key in targets:
            results[key] = torch.stack(targets[key], dim=0).to(device)
    else:
        raise NotImplementedError
    return results


def make_targets(cfg, inputs, device):
    targets = inputs['labels'].to(device)
    return targets

def get_idx_from_score_by_threshold(threshold=0.5, seq_indices=None, seq_scores=None):
    seq_indices = np.array(seq_indices)
    seq_scores = np.array(seq_scores)
    bdy_indices = []
    internals_indices = []
    for i in range(len(seq_scores)):
        if seq_scores[i] >= threshold:
            internals_indices.append(i)
        elif seq_scores[i] < threshold and len(internals_indices) != 0:
            bdy_indices.append(internals_indices)
            internals_indices = []

        if i == len(seq_scores) - 1 and len(internals_indices) != 0:
            bdy_indices.append(internals_indices)

    bdy_indices_in_video = []
    if len(bdy_indices) != 0:
        for internals in bdy_indices:
            center = int(np.mean(internals))
            bdy_indices_in_video.append(seq_indices[center])
    return bdy_indices_in_video

@torch.no_grad()
def main(cfg, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(cfg)
    model.eval()

    #**************************************************
    #If test the GFLOPS, uncomment the following lines

    # from fvcore.nn import FlopCountAnalysis
    # x = torch.rand(1,100,3,224,224).cuda()
    # model.eval()
    # flops = FlopCountAnalysis(model, x)
    # print("FLOPs: ", flops.total() / 1e9)
    # return

    #**************************************************

    if args.resume:
        state_dict = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(state_dict['model'], strict=False)
    model = model.to(device)
    input_size = (cfg['INPUT']['RESOLUTION'], cfg['INPUT']['RESOLUTION'])
    # transform = [transforms.Resize(input_size)]
    transform = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]
    val_transform = transforms.Compose(transform)
    model_name = os.path.dirname(args.resume).split('/')[-1]
    threshold = args.threshold
    # visualize = True

    annotationfile_path = args.annotationfile_path

    video_dir = args.video_dir
    base_dir = os.path.dirname(os.path.dirname(__file__))
    if 'xd_violence' in annotationfile_path:
        # video_dir ="/root/autodl-fs/lwl/data/XD_test/test_videos/"
        out_path = f'{base_dir}/result/XD_Violence_test/EGEBD_{model_name}_split_th{threshold}'
    elif 'ucf_crime' in annotationfile_path:
        # video_dir ="/root/autodl-fs/lwl/data/UCF_Crime_test/test_videos"
        out_path = f'{base_dir}/result/UCF_Crime_test/EGEBD_{model_name}_split_out_th{threshold}/'
    elif 'MSAD' in annotationfile_path:
        # video_dir ="/root/autodl-fs/lwl/data/MSAD_test/test_videos"
        out_path = f'{base_dir}/result/MSAD_test/EGEBD_{model_name}_split_out_th{threshold}/'
    else:
        raise ValueError(f'Unknown dataset in annotationfile_path: {annotationfile_path}')

    # if '103' in annotationfile_path:
    #     out_path += '_sub103'
    print('out_path:', os.path.abspath(out_path))

    os.makedirs(out_path, exist_ok=True)
    if args.visualize:
        out_vis_path = os.path.join(out_path, 'split_vis')
        os.makedirs(out_vis_path, exist_ok=True)


    # load data & build fnm - path dict
    fnm_path_dict = {}

    video_list_ = [os.path.basename(x.strip().split()[0]) for x in open(annotationfile_path)]
    video_list_ = sorted(
        video_list_,
        key=lambda x:
        # x.path # 元组排序规则：(是否包含Normal, 键本身)
        ("ormal" in x or "label_A" in x, x)  # 元组排序规则：(是否包含Normal/normal, 键本身)
    )
    for i in video_list_:
        fnm_path_dict[i+'.mp4'] = os.path.join(video_dir, i+'.mp4')

    print(len(fnm_path_dict.keys()))

    all_pred_scenes = {}
    all_scenes = {}
    i = 0
    for name, vid_path in fnm_path_dict.items():
        i += 1


        predictions = []

        frames_split_100, org_frames_split_100, vid_len, idx_split_100, fps = get_split_100frames_opencv_trans(vid_path,(input_size[0],input_size[1],3),val_transform)
        print(i, datetime.now().strftime("%H:%M:%S"), vid_len, name)

        predictions = []
        for batch in get_batches(frames_split_100):
            model_input = {'imgs':batch[None,...].to(device).to(torch.float32)}
            model_out = model(model_input)
            model_out = model_out.detach().cpu().numpy()
            predictions.append(model_out[0,0,25:75])

        predictions = np.concatenate(predictions, 0)[:len(idx_split_100)]

        out_idx = get_idx_from_score_by_threshold(threshold=threshold, seq_indices=idx_split_100, seq_scores=predictions)
        print(out_idx)
        scences = []
        s = 0
        for j in out_idx:
            j = int(j)
            scences.append([s,j])
            s = j
        scences.append([s, vid_len-1])

        predictions = np.around(predictions, 2)
        predictions = [round(float(i),2) for i in predictions]
        all_pred_scenes[name] = {'pred':predictions,'scenes':scences, 'fps':fps ,'frames':int(
            vid_len)}
        all_scenes[name] = {'scenes':scences, 'fps':fps ,'frames':int(vid_len)}
        print(f'\t {len(scences)} boundrary: {np.array(scences)[:,1] - np.array(scences)[:,0]}')
        print(f'\t {len(scences)} span seconds: {np.round((np.array(scences)[:,1] - np.array(scences)[:,0])/fps,1)}')
        if args.visualize:
            pil_image = visualize_predictions_TransNetv2(
                org_frames_split_100, predictions=(model_out[0,0],model_out[0,0]))
            pil_image.save(os.path.join(out_vis_path, name) + ".vis.png")

        # 保存输出文档
        with open(os.path.join(out_path, f'pred_scenes_th{threshold}.json'), 'w', encoding='utf-8') as json_file:
            json.dump(all_pred_scenes, json_file, ensure_ascii=False, indent=4)
        with open(os.path.join(out_path, f'scenes_th{threshold}.json'), 'w', encoding='utf-8') as json_file:
            json.dump(all_scenes, json_file, ensure_ascii=False, indent=4)



def init_seeds(seed, cuda_deterministic=True):
    cudnn.enabled = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_dir", type=str,
        # default='/root/autodl-fs/lwl/data/XD_test/test_videos/'
        default='/root/autodl-fs/lwl/data/UCF_Crime_test/test_videos',
        # default='/root/autodl-fs/lwl/data/MSAD_test/test_videos'
        help='Path to the directory containing videos.'
    )
    parser.add_argument("--annotationfile_path", type=str,
        default=None,
        help='Path to the annotation file for evaluation, if None, auto set according to video_dir'
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--config-file", type=str,
    default='/root/autodl-tmp/EfficientGEBD/config-files/baseline.yaml',
    # default='/root/autodl-tmp/EfficientGEBD/data/x1_r50_basic/scripts/baseline_end_to_end_diff_former.yaml'
                        )
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--resume", type=str,
                        # default='/root/autodl-tmp/EfficientGEBD/data/x1_r50_basic/model_best.pth',
                        # default='/root/autodl-tmp/EfficientGEBD/data/x2x3x4_r18_eff/model_best.pth',
                        default='/root/autodl-tmp/EfficientGEBD/data/x2x3x4_r50_eff/model_best.pth',
                        )
    parser.add_argument("--visualize", default=False, action='store_true', help='whether to visualize the scene segmentation results')
    parser.add_argument("--test-only", action='store_true', default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expname", type=str, default='test')
    parser.add_argument("--all-thres", action='store_true', default=True, help='test using all thresholds [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]')

    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()


    if args.annotationfile_path is not None: # 自动设置annotationfile_path
        vadtree_path = os.path.dirname(os.path.dirname(__file__))
        if 'XD' in args.video_dir:
            args.annotationfile_path=f'{vadtree_path}/dataset_info/xd_violence/annotations/anomaly_test.txt'
        elif 'UCF' in args.video_dir:
            args.annotationfile_path = f'{vadtree_path}/dataset_info/ucf_crime/annotations/anomaly_test.txt'    
        elif 'MSAD' in args.video_dir:
            args.annotationfile_path=f'{vadtree_path}/dataset_info/MSAD_test/anomaly_test.txt'
        else:
            raise ValueError(f'Unknown dataset in video_dir: {args.video_dir}')

    if not args.local_rank:
        args.local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    args.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = args.num_gpus > 1

    if torch.cuda.is_available():
        # torch.backends.cudnn.benchmark = True
        init_seeds(args.seed + args.local_rank)
        if args.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend="nccl", world_size=args.num_gpus, rank=args.local_rank)
            dist.barrier()
    cfg.merge_from_file(args.config_file)
    

    # 根据不同模型修改cfg
    if 'x1_r50_basic' in args.resume:
        cfg.MODEL.BACKBONE.NAME = 'resnet50'
        cfg.MODEL.CAT_PREV = False
        cfg.MODEL.FPN_START_IDX = 0
        cfg.MODEL.HEAD_CHOICE = [0]
        cfg.MODEL.IS_BASIC = True
    elif 'x2x3x4_r18_eff' in args.resume:
        ################ x2x3x4_r18_eff
        cfg.MODEL.BACKBONE.NAME = 'resnet18'
        cfg.MODEL.CAT_PREV = True
        cfg.MODEL.FPN_START_IDX = 1
        cfg.MODEL.HEAD_CHOICE = [3]
        cfg.MODEL.IS_BASIC = False
    elif 'x2x3x4_r50_eff' in args.resume:
        ################ x2x3x4_r50_eff
        cfg.MODEL.BACKBONE.NAME = 'resnet50'
        cfg.MODEL.CAT_PREV = True
        cfg.MODEL.FPN_START_IDX = 1
        cfg.MODEL.HEAD_CHOICE = [3]
        cfg.MODEL.IS_BASIC = False

    cfg.freeze()
    # print(cfg.MODEL)
    if is_main_process():
        print('Args: \n{}'.format(args))
        print('Configs: \n{}'.format(cfg))

    main(cfg, args)
