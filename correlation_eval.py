import argparse, glob
import numpy as np, re

from src.data.video_record import VideoRecord
from src.utils.vis_utils import visualize_video
from src.utils.eval_utils import *
from src.utils.ensemble_utils import *

def node_sort(coarse_scores_json, fine_scores_json):
    # 使用正则表达式精准定位
    match0 = re.search(r"_dfs_([0-9.]+)_(\d+)", coarse_scores_json)
    match1 = re.search(r"_dfs_([0-9.]+)_(\d+)", fine_scores_json)

    if float(match0.group(1)) > float(match1.group(1)): # 阈值高的为父节点
        print('root node coarse_scores_json：', coarse_scores_json)
        print('child node coarse_scores_json：', fine_scores_json)
        return coarse_scores_json, fine_scores_json
    elif float(match0.group(1)) == float(match1.group(1)):  # 阈值相同，则最小片段长度更大的为父节点
        if float(match0.group(2)) >= float(match1.group(2)):
            print('root node coarse_scores_json：', coarse_scores_json)
            print('child node coarse_scores_json：', fine_scores_json)
            return coarse_scores_json, fine_scores_json

    print('root node coarse_scores_json：', fine_scores_json)
    print('child node coarse_scores_json：', coarse_scores_json)
    return fine_scores_json, coarse_scores_json # 其他情况

def parse_args():

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--video_root", type=str,
        default=None,
        help="Root directory containing the video files. You can set it to None if not visualize or auto set according to dataset."
    )

    # UCF-Crime
    parser.add_argument("--coarse_scores_json", type=str,
        default='result/UCF_Crime_test/EGEBD_x2x3x4_r50_eff_split_out_th0.5_peak_dfs_kmeans_1_0.4/LLaVA-Video-7B-Qwen2_ucf_prior_q_coarse/DeepSeek-R1-Distill-Qwen-14B_think_VxV10_nn10_tao0.1/refine_maxf64_ucf_prior_q_Here is a .json',
    )
    parser.add_argument("--fine_scores_json", type=str,
        default='result/UCF_Crime_test/EGEBD_x2x3x4_r50_eff_split_out_th0.5_peak_dfs_kmeans_1_0.4/LLaVA-Video-7B-Qwen2_ucf_prior_q_fine/DeepSeek-R1-Distill-Qwen-14B_think_VxV10_nn10_tao0.1/refine_maxf64_ucf_prior_q_Here is a .json',
    )

    parser.add_argument("--beta",
        type=float,
        default=0.2,
        # default=-0.6,
        help="beta for score correlation",
    )
    parser.add_argument("--without_labels", default=False, action="store_true")
    parser.add_argument("--visualize",
        # default=True,
        default=False,
        action="store_true")

    args = parser.parse_args()


    #  自动化配置参数补全
    vadtree_path = os.path.dirname(__file__)
    args.vadtree_path = vadtree_path
    if 'UCF' in args.coarse_scores_json:
        assert 'UCF' in args.fine_scores_json
        args.normal_label = 7
        args.video_fps = 30.0
        args.annotationfile_path=f'{vadtree_path}/dataset_info/ucf_crime/annotations/anomaly_test.txt'
        args.temporal_annotation_file=f'{vadtree_path}/dataset_info/ucf_crime/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt'
        if args.video_root is None: args.video_root = "/root/autodl-fs/lwl/data/UCF_Crime_test/test_videos"

    elif 'XD' in args.coarse_scores_json:
        assert 'XD' in args.fine_scores_json
        args.normal_label = 4
        args.video_fps = 24.0
        args.annotationfile_path=f'{vadtree_path}/dataset_info/xd_violence/annotations/anomaly_test.txt'
        args.temporal_annotation_file=f'{vadtree_path}/dataset_info/xd_violence/annotations/temporal_anomaly_annotation_for_testing_videos.txt'
        if args.video_root is None: args.video_root = "/root/autodl-fs/lwl/data/XD_test/test_videos/"

    elif 'MSAD' in args.coarse_scores_json:
        args.normal_label = 0
        args.video_fps = 'auto'
        args.annotationfile_path=f'{vadtree_path}/dataset_info/MSAD_test/anomaly_test.txt'
        args.temporal_annotation_file=f'{vadtree_path}/dataset_info/MSAD_test/Temporal_Anomaly_Annotation_for_Testing_Videos.txt'
        if args.video_root is None: args.video_root = "/root/autodl-fs/lwl/data/MSAD_test/test_videos/"
    
    args.output_dir = os.path.dirname(args.coarse_scores_json)

    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(args.coarse_scores_json)))
    if 'split_mean' in args.coarse_scores_json: # 均匀采样的实验
        pattern = os.path.join(parent_dir, "scenes.json")
    else:
        pattern = os.path.join(parent_dir, "pred.json")
    candidates = glob.glob(pattern)
    assert len(candidates)>=1
    args.split_pred = candidates[0]

    # 集成后的分数存储
    if args.fine_scores_json!=None:
        args.output_dir += f'_ENSE'
        name = os.path.normpath(os.path.dirname(args.fine_scores_json)).replace('EGEBD_x2x3x4_r50_eff_split_out_th',
            'EX234R50ES')
        name =name.replace('LLaVA-Video-7B-Qwen2', 'LV7Q').replace('DeepSeek-R1-Distill-Qwen-14B', 'DRDQ14')
        name = '_'.join(name.split('/')[-3:])
        args.output_dir += f'_{name}_beat{args.beta}'
        args.ense_score_output_json = os.path.join(args.output_dir, 'ense_'+os.path.basename(args.coarse_scores_json))


    return args



def main(
        args
):

    # Load the temporal annotations
    if not args.without_labels:
        annotations = temporal_testing_annotations(args.temporal_annotation_file)

    # Load video records from the annotation file
    video_list = [VideoRecord(x.strip().split(), '' if args.video_root is None else args.video_root) for x in open(
        args.annotationfile_path)]

    flat_scores, flat_labels = initialize_score_dicts(
        scores_json=args.coarse_scores_json
    )

    vid_metric = {}
    dataset_metric = {}

    # load all_video_scores_dict
    with open(args.coarse_scores_json) as f:
        all_video_scores_dict = json.load(f)
    # pprint(all_video_scores_dict['config'])
    out_all_video_scores_dict = copy.deepcopy(all_video_scores_dict)

    # load ense_scores_dict
    with open(args.fine_scores_json) as f:
        all_ense_scores_dict = json.load(f)

    if args.split_pred is not None:
        with open(args.split_pred) as f:
            all_video_split_pred = json.load(f)

    if 'MSAD' in args.coarse_scores_json: # load video info for fps
        with open(f'{args.vadtree_path}/MSAD_test/EGEBD_x2x3x4_r50_eff_split_out_th0.5_peak_dfs_kmeans_1_0.4'
                  '/dfs_coarse_sences.json') as f:
            msad_video_info = json.load(f)

    ab_pro_mean = []
    ab_pro_sum = []
    ab_pro_var = []
    ab_pro_max = []
    ab_max = []
    all_ab_s = []
    ab_s = []
    ab_f_pro = []
    all_ab_f_pro = []
    all_ab_pro_mean = []
    for k, video in enumerate(video_list):

        video_name = Path(video.path).name
        # Get video labels
        if args.without_labels:
            video_labels = []
        else:
            video_labels = get_video_labels(video, annotations, args.normal_label)

        if args.video_fps=='auto':
            video_fps = msad_video_info[video_name+'.mp4']['fps']
        # Load the scores
        video_scores_dict = all_video_scores_dict['vid_score'][video_name + '.mp4']
        ense_video_scores_dict = all_ense_scores_dict['vid_score'][video_name + '.mp4']
        video_scores = [0.0] * len(video_labels)
        video_scores_ = [0.0] * len(video_labels)
        # video_captions = all_video_caption_dict['vid_captions'][video_name + '.mp4']
        video_captions = [video_scores_dict, ense_video_scores_dict]

        video_variances = compute_normalized_variances(
            video_scores_dict, ense_video_scores_dict,
            normalization='minmax',
            # 'softmax'
        )
        # weight = [0.5, 0.5]
        weight_list = [0.5] * len(video_labels) #原始分数的权重。
        scenes = []
        vid_ab_pro = []
        for idx, (clip, score) in enumerate(video_scores_dict.items()):
            start, end = map(lambda x: int(float(x)), clip.split(', '))
            scenes += [start, end]
            if isinstance(score, list) and 'think' in args.coarse_scores_json: # 处理think信息
                think_info = score[-1]
                if 'dur' in args.coarse_scores_json:  # 处理dur信息
                    dur = score[1]
                    start, end = dur_refine(start, end, dur)
            score_ = score[1] # 未经refine的分数
            score = score[0]
            video_scores[start:end+1] = [score] * (end - start+1)
            # video_scores_org[start:end+1] = [score_] * (end - start+1)
            video_scores_[start:end+1] = [score_] * (end - start+1)
            vid_ab_pro.append(((end-start)/len(video_labels))* score_)
            # fine_w = 0

        all_ab_s.append(sum(video_scores)/len(video_labels))
        for idx, (clip, score) in enumerate(video_scores_dict.items()):
            start, end = map(lambda x: int(float(x)), clip.split(', '))
            fine_w = video_variances[clip]['norm_score']*args.beta
            weight_list[start:end+1] = [0.5 - fine_w/2] * (end - start+1)

        print(f'{video_name} vid_ab_pro', np.array(vid_ab_pro).mean(),  np.array(vid_ab_pro).sum(),
            np.array(vid_ab_pro).max(),vid_ab_pro)
        # if args.normal_label not in video_labels:

        all_ab_f_pro.append(np.array(np.array(video_labels)!= args.normal_label).sum()/len(video_labels)) # 计算异常片段的比例
        all_ab_pro_mean.append(np.array(vid_ab_pro).mean())

        if len(set(video_labels))==1 and  args.normal_label in set(video_labels): #不含有异常片段
        # if 0: #不含有异常片段
            print()
        else:
            ab_s.append(sum(video_scores_)/len(video_labels))
            ab_pro_mean.append(np.array(vid_ab_pro).mean())
            ab_pro_sum.append(np.array(vid_ab_pro).sum())
            ab_pro_var.append(np.array(vid_ab_pro).var())
            ab_pro_max.append(np.array(vid_ab_pro).max())
            ab_max.append(np.array(vid_ab_pro).max()*len(video_labels))
        # 待集成的分数
        ense_scores = get_video_flat_scores(ense_video_scores_dict) # 待集成的分数

        # 加权
        # 兼容了MSAD数据集中标签长度与实际视频帧数量不匹配的情况
        w_video_scores = np.array(weight_list)* np.array(video_scores)
        w_ense_scores = (1-np.array(weight_list)[:len(ense_scores)])*np.array(ense_scores)
        w_video_scores[:len(w_ense_scores)] += w_ense_scores
        ense_video_scores = list(w_video_scores)

        ense_video_scores =  [round(num,4)  for num in ense_video_scores]


        assert len(ense_video_scores) == video.num_frames

        out_all_video_scores_dict['vid_score'][video_name + '.mp4'] = ense_video_scores

        # Extend scores and labels
        flat_scores, flat_labels = update_flat_scores_labels(
            flat_scores, flat_labels, ense_video_scores, video_labels, args.without_labels, args.coarse_scores_json, video_name
        )

        # 计算单个视频的指标
        vid_binary_labels = np.array(video_labels) != args.normal_label

        fpr, tpr, threshold = roc_curve(vid_binary_labels, np.array(ense_video_scores))
        roc_auc = auc(fpr, tpr)

        # Compute precision-recall curve
        precision, recall, th = precision_recall_curve(vid_binary_labels, np.array(ense_video_scores))
        pr_auc = auc(recall, precision)

        pos_mean = round(np.array(ense_video_scores)[vid_binary_labels].mean(),4)
        neg_mean = round(np.array(ense_video_scores)[~vid_binary_labels].mean(),4)
        vid_metric[video_name] = {'roc_auc':round(roc_auc,4), 'pr_auc':round(pr_auc,4), '1_mean':pos_mean,
            '0_mean':neg_mean}
        print(str(k).zfill(4), video_name, f' {vid_metric[video_name]} ')

        # if pr_auc>0.8:
        #     print('sdadadaedwawd',pr_auc, roc_auc)

        # if False:
        if args.visualize:
            vis_dir = os.path.join(args.output_dir, 'vis_dir')
            os.makedirs(vis_dir, exist_ok=True)

            # visualize_video
            visualize_video(
                video_name,
                video_labels if not args.without_labels else [],
                ense_video_scores,
                video_captions,
                args.video_root,
                video_fps if args.video_fps == 'auto' else args.video_fps,
                Path(vis_dir) / f"{video_name}.mp4",
                args.normal_label,
                "{:06d}.jpg",
                save_video=False,
                # video_metric = f'roc_auc:{round(roc_auc,4)}, pr_auc:{round(pr_auc,4)},1_mean:{pos_mean},  '
                #                f'0_mean:{neg_mean}',
                scenes=scenes,
                # scenes=None,
                # split_pred=all_video_split_pred[video_name + '.mp4'] if args.split_pred is not None else None,
                # split_pred=None,
            )

    if not args.without_labels:
        dataset_metric = compute_metrics(flat_scores, flat_labels, args.normal_label)

        out_vid_metric = {'vid_metric':vid_metric,
            'dataset_metric':dataset_metric,
            'args_dict': args_dict,
        }
        # 每个视频的结果汇聚为一个json
        print(json.dumps(dataset_metric, indent=4, ensure_ascii=False))
        print(json.dumps(dataset_metric['all'], indent=4, ensure_ascii=False))

        if len(video_list) not in [800, 103, 140, 290, 240]:
            print('!!!!!!!!!!!!!val mode, not write!!!!!!!!!!!!')

        # print('!!!!!!!!!!!!!val mode, not write!!!!!!!!!!!!')

        # some statistics results
        # print('ab_pro_mean',np.array(ab_pro_mean).mean(), ab_pro_mean)
        # print('ab_pro_sum',np.array(ab_pro_sum).mean(), ab_pro_sum)
        # print('ab_pro_var',np.array(ab_pro_var).mean(), ab_pro_var)
        # print('ab_pro_max',np.array(ab_pro_max).mean(), ab_pro_max)
        # print('ab_max',np.array(ab_max).mean(), ab_max)
        # print('all_video_all_ab_s',np.array(all_ab_s).mean(),all_ab_s)
        # print('ab_s',np.array(ab_s).mean(),ab_s)
        # print('ab_f_pro', np.array(ab_f_pro).mean(), ab_f_pro)
        # print('all_ab_f_pro', np.array(all_ab_f_pro).mean())
        # print('all_ab_pro_mean', np.array(all_ab_pro_mean).mean())

        # sys.exit(0)
        os.makedirs(args.output_dir, exist_ok=True)


        out_json_name = '00vid_metric' + os.path.basename(args.annotationfile_path)[:-4].replace('anomaly_test', '')
        with open(f'{args.output_dir}/{out_json_name}.json', 'w', encoding='utf-8') as json_file:
            # 序列化数据到文件，设置 ensure_ascii=False 以支持非ASCII字符
            # indent 参数用于格式化输出，使 JSON 文件更易读
            json.dump(out_vid_metric, json_file, ensure_ascii=False, indent=4)
        print(f'\nsave path:{args.output_dir}/00vid_metric.json')

        # refine后得分存储
        del out_all_video_scores_dict['config']
        out_all_video_scores_dict['dataset_metric'] = dataset_metric
        out_all_video_scores_dict['args_dict'] = args_dict
        with open(args.ense_score_output_json, 'w', encoding='utf-8') as json_file:
            json.dump(out_all_video_scores_dict, json_file, ensure_ascii=False, indent=4)
        print(f'\nout_all_video_scores_dict save path:{args.ense_score_output_json}')

if __name__ == "__main__":
    args = parse_args()
    # 转换并序列化
    args_dict = args_to_dict(args,max_depth=5)
    print('-'*50)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print('-'*50)
    main(
        args
    )

