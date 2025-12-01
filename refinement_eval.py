import argparse, glob
import os, pickle
from src.data.video_record import VideoRecord
from src.utils.vis_utils import visualize_video
from src.utils.eval_utils import *

def parse_args():
    '''
    '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--scores_json", type=str,
        default='result/UCF_Crime_test/EGEBD_x2x3x4_r50_eff_split_out_th0.5_peak_dfs_kmeans_1_0.4/LLaVA-Video-7B-Qwen2_ucf_prior_q_coarse/DeepSeek-R1-Distill-Qwen-14B_think/maxf64_ucf_prior_q_Here is a .json',
    )
    parser.add_argument("--video_root", type=str,
        default=None,
        help="Root directory containing the video files. You can set it to None if not visualize or auto set according to dataset."
    )
    parser.add_argument("--similarity_pkl", type=bool,
        default=True,
        # default=False,
        help='Whether to execute intra-cluster node refinement. If True, the similarity data will automatically set '
             'according to scores_json path.'
    )

    parser.add_argument("--similarity_type", type=str,
        # default='VxT', # V: visual feature, T: text feature
        # default='TxT',
        # default='TxV',
        default='VxV', 
        # default=[[0.5,'VxV'],[0.5,'VxT']], # weighted sum of multiple similarity types
        # default=[[0.5,'VxV'],[0.5,'TxT']],
        # default=[[0.7,'VxV'],[0.3,'VxT']],
        # default=['VxT', 'TxV', 'VxV','TxT' ],
        help='Type of similarity to use for score refinement. Options include VxV (visual feature x visual feature), VxT (visual feature x text feature), TxT (text feature x text feature), and TxV (text feature x visual feature). You can also provide a weighted sum of multiple similarity types as a list of tuples.',
    )
    parser.add_argument("--topK", type=int, default=10)
    parser.add_argument("--num_neighbors", type=int, default=10)
    parser.add_argument("--dyn_ratio", type=float,
        # default=0.2,
        default=None,
        help='dynamic ratio for determining number of neighbors in score refinement.'
    ) # topK和nn的动态一致取值，优先级最高（非None时）, nn取 max（比例， nn参数）
    parser.add_argument("--tao", type=float,
        # default=1,
        default=0.1,
        help='Temperature for refining scores, lower tao leads to more winner-takes-all behavior.'
    ) # refine分数时的温度，tao越低，赢者通吃

    parser.add_argument("--without_labels", default=False, action="store_true")
    parser.add_argument("--visualize",
        # default=True,
        default=False,
        action="store_true",
        help="Whether to visualize the results."
        )

    args = parser.parse_args()

    # auto set args according to dataset
    vadtree_path = os.path.dirname(__file__)
    if args.video_root is None:
        if 'UCF' in args.scores_json:
            args.normal_label = 7
            args.video_fps = 30.0
            args.annotationfile_path=f'{vadtree_path}/dataset_info/ucf_crime/annotations/anomaly_test.txt'
            args.temporal_annotation_file=f'{vadtree_path}/dataset_info/ucf_crime/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt'
            if args.video_root is None: args.video_root = "/root/autodl-fs/lwl/data/UCF_Crime_test/test_videos"
        elif 'XD' in args.scores_json:
            args.normal_label = 4
            args.video_fps = 24.0
            args.annotationfile_path=f'{vadtree_path}/dataset_info/xd_violence/annotations/anomaly_test.txt'
            args.temporal_annotation_file=f'{vadtree_path}/dataset_info/xd_violence/annotations/temporal_anomaly_annotation_for_testing_videos.txt'
            if args.video_root is None: args.video_root = "/root/autodl-fs/lwl/data/XD_test/test_videos/"
        elif 'MSAD' in args.scores_json:
            args.normal_label = 0
            args.video_fps = 'auto' # MSAD数据集视频帧率不统一，自动获取
            args.annotationfile_path=f'{vadtree_path}/dataset_info/msad/annotations/anomaly_test.txt'
            args.temporal_annotation_file=f'{vadtree_path}/dataset_info/msad/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt'
            if args.video_root is None: args.video_root = "/root/autodl-fs/lwl/data/MSAD_test/test_videos/"

    args.output_dir = os.path.dirname(args.scores_json)

    if '_split' in args.scores_json: # 基于dnn生成的有pred的边界分数的实验
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(args.scores_json)))
        pattern = os.path.join(parent_dir, "pred.json")
        candidates = glob.glob(pattern)
        # assert len(candidates)==1
        args.split_pred = candidates[0]

    if args.similarity_pkl!=False:
        if type(args.similarity_pkl) != str:  # 设定sim pkl路径
            args.similarity_pkl = os.path.join(os.path.dirname(os.path.dirname(args.scores_json)), 'sim_' + os.path.basename(
                args.scores_json)[:-4]+'pkl')
        assert(os.path.exists(args.similarity_pkl))
        print(f'---------------------{args.similarity_type} similarity_pkl: {args.similarity_pkl}-------------------------------')

        if isinstance(args.similarity_type, str): # 设定输出路径
            args.output_dir += f'_{args.similarity_type}{args.topK}_nn{args.num_neighbors}'
        elif isinstance(args.similarity_type, list):
            for i in args.similarity_type:
                args.output_dir += f'_{i[0]}{i[1]}{args.topK}_nn{args.num_neighbors}'
        else:
            raise NotImplementedError
        if 'sum1' in args.similarity_pkl:
            args.output_dir += '_sum1'

        if args.dyn_ratio!=None:
            args.output_dir += f'_dyn_r{args.dyn_ratio}'
        if args.tao != 1:
            args.output_dir += f'_tao{args.tao}'

        if args.temporal_annotation_file is None and not args.without_labels:
            parser.error("--temporal_annotation_file is required when --without_labels is not used")
        if args.visualize:
            if args.video_fps is None:
                parser.error("--video_fps is required when --visualize is used")

    # refine后的分数存储
    args.refine_score_output_json = os.path.join(args.output_dir, 'refine_'+os.path.basename(args.scores_json))


    return args


def main(
        video_root,
        annotationfile_path,
        temporal_annotation_file,
        scores_json,
        similarity_pkl,
        output_dir,
        normal_label,
        topK,
        without_labels,
        visualize,
        video_fps,
):
    # Convert path
    scores_json = Path(scores_json)
    # similarity_pkl = similarity_pkl


    # Load the temporal annotations
    if not without_labels:
        annotations = temporal_testing_annotations(temporal_annotation_file)

    # Load video records from the annotation file
    video_list = [VideoRecord(x.strip().split(), '' if args.video_root is None else args.video_root) for x in open(
        args.annotationfile_path)]


    flat_scores, flat_labels = initialize_score_dicts(
        scores_json=args.scores_json
    )

    vid_metric = {}
    dataset_metric = {}

    # load all_video_scores_dict
    with open(scores_json) as f:
        all_video_scores_dict = json.load(f)
    out_all_video_scores_dict = copy.deepcopy(all_video_scores_dict)

    if args.similarity_pkl!=False:
        with open(similarity_pkl, "rb") as f:
            similarity_dict = pickle.load(f)

    if '_split' in args.scores_json:
        if args.split_pred is not None:
            with open(args.split_pred) as f:
                all_video_split_pred = json.load(f)

    if 'MSAD' in args.scores_json: # load video info for fps
            with open(f'{args.vadtree_path}/MSAD_test/EGEBD_x2x3x4_r50_eff_split_out_th0.5_peak_dfs_kmeans_1_0.4'
                      '/dfs_coarse_sences.json') as f:
                msad_video_info = json.load(f)

    for k, video in enumerate(video_list):
        # if 'Bad.Boys.1995__#01-11-55_01-12-40_label_G-B2-B6' not in video.path:
        #     continue

        video_name = Path(video.path).name
        # Get video labels
        if without_labels:
            video_labels = []
        else:
            video_labels = get_video_labels(video, annotations, normal_label)


        # Load the scores and similarity
        video_scores_dict = all_video_scores_dict['vid_score'][video_name + '.mp4']
        video_scores = [0.0] * len(video_labels)

        video_captions = None

        if args.similarity_pkl!=False:  # socore refine
            vodeo_similarity_dict = similarity_dict['vid_sim'][video_name + '.mp4']
            video_scores_dict = calculate_refine_scores(
                video_scores_dict, vodeo_similarity_dict, args.similarity_type, args.topK,
                args.num_neighbors, args.dyn_ratio,  args.tao, args
            )
            for m, n in video_scores_dict.items(): # 记录refine后分数
                if type(out_all_video_scores_dict['vid_score'][video_name + '.mp4'][m]) is list:
                    out_all_video_scores_dict['vid_score'][video_name + '.mp4'][m].insert(0, round(n,4))
                else:
                    out_all_video_scores_dict['vid_score'][video_name + '.mp4'][m] = [out_all_video_scores_dict[
                        'vid_score'][video_name +
                                     '.mp4'][m], round(n,4)]

        if type(video_scores_dict) is list:
            video_scores = video_scores_dict
        else:
            scenes = []
            for idx, (clip, score) in enumerate(video_scores_dict.items()):
                start, end = map(lambda x: int(float(x)), clip.split(', '))
                scenes += [start, end]
                if isinstance(score, list):
                    think_info = score[-1]
                    if 'dur' in args.scores_json:
                        dur = score[1]
                        start, end = dur_refine(start, end, dur)
                    score = score[0]
                video_scores[start:end+1] = [score] * (end - start+1)

        assert len(video_scores) == video.num_frames
        video_scores = video_scores[: video.num_frames]

        video_scores=1/(1+np.exp(-np.array(video_scores)) )
        video_scores =  [ round(num,4)  for num in video_scores]
        # Extend scores and labels
        flat_scores, flat_labels = update_flat_scores_labels(
            flat_scores, flat_labels, video_scores, video_labels, without_labels, args.scores_json, video_name
        )

        # 计算单个视频的指标
        if not without_labels:
            vid_binary_labels = np.array(video_labels) != normal_label
            # Compute ROC AUC score

            fpr, tpr, threshold = roc_curve(vid_binary_labels, np.array(video_scores))
            roc_auc = auc(fpr, tpr)

            # Compute precision-recall curve
            precision, recall, th = precision_recall_curve(vid_binary_labels, np.array(video_scores))
            pr_auc = auc(recall, precision)

            if np.array(vid_binary_labels, dtype=bool).any():
                pos_mean = round(np.array(video_scores)[vid_binary_labels].mean(),4)
            else:
                pos_mean = 0.
            neg_mean = round(np.array(video_scores)[~vid_binary_labels].mean(),4)
            vid_metric[video_name] = {'roc_auc':round(roc_auc,4), 'pr_auc':round(pr_auc,4), '1_mean':pos_mean,
                '0_mean':neg_mean}
            print(str(k).zfill(4), video_name, f' {vid_metric[video_name]} ')

        if args.video_fps=='auto':
            video_fps = msad_video_info[video_name+'.mp4']['fps']
        if visualize:
            vis_dir = os.path.join(output_dir, 'vis_dir')
            os.makedirs(vis_dir, exist_ok=True)

            # visualize_video
            visualize_video(
                video_name,
                video_labels if not args.without_labels else [],
                video_scores,
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

    if not without_labels:
        dataset_metric = compute_metrics(flat_scores, flat_labels, normal_label)

        out_vid_metric = {'vid_metric':vid_metric,
            'dataset_metric':dataset_metric,
            'args_dict': args_dict,
        }
        # 每个视频的结果汇聚为一个json
        print(json.dumps(dataset_metric, indent=4, ensure_ascii=False))
        print(json.dumps(dataset_metric['all'], indent=4, ensure_ascii=False))
        # print('dataset_metric :', dataset_metric)

        if len(video_list) not in [800, 103, 140, 290]:
            print('!!!!!!!!!!!!!val mode, not write!!!!!!!!!!!!')

        # print('!!!!!!!!!!!!!val mode, not write!!!!!!!!!!!!')
        # sys.exit(0)

        os.makedirs(args.output_dir, exist_ok=True)

        out_json_name = '00vid_metric' + os.path.basename(args.annotationfile_path)[:-4].replace('anomaly_test', '')
        with open(f'{output_dir}/{out_json_name}.json', 'w', encoding='utf-8') as json_file:
            # 序列化数据到文件，设置 ensure_ascii=False 以支持非ASCII字符
            # indent 参数用于格式化输出，使 JSON 文件更易读
            json.dump(out_vid_metric, json_file, ensure_ascii=False, indent=4)
        print(f'\nsave path:{os.path.abspath(output_dir)}/00vid_metric.json')

        # refine后得分存储
        if args.similarity_pkl != False:
            # if not os.path.exists(args.refine_score_output_json):
            out_all_video_scores_dict['args_dict'] = args_dict
            out_all_video_scores_dict['dataset_metric'] = dataset_metric
            with open(args.refine_score_output_json, 'w', encoding='utf-8') as json_file:
                json.dump(out_all_video_scores_dict, json_file, ensure_ascii=False, indent=4)
            # print(f'\nout_all_video_scores_dict save path:{args.refine_score_output_json}')
            print(f'\nout_all_video_scores_dict:{os.path.abspath(args.refine_score_output_json)}')


if __name__ == "__main__":
    args = parse_args()
    # 转换并序列化
    args_dict = args_to_dict(args,max_depth=5)
    print('-'*50)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print('-'*50)
    main(
        args.video_root,
        args.annotationfile_path,
        args.temporal_annotation_file,
        args.scores_json,
        args.similarity_pkl,
        args.output_dir,
        args.normal_label,
        args.topK,
        args.without_labels,
        args.visualize,
        args.video_fps,
    )
