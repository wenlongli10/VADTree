import copy
import os,sys
import argparse
import numpy as np
import json
from pathlib import Path
# import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# 数据集类别定义
UCF_Crime_classes = ['Normal', 'Abuse', 'Arrest', 'Arson','Assault', 'Burglary',
    'Explosion', 'Fighting', 'RoadAccidents', 'Robbery',
    'Shooting','Shoplifting','Stealing','Vandalism', ]

XD_classes = [ 'Normal', 'Fighting', 'Shooting', 'Riot', 'Abuse','Car Accident',
    'Explosion']
XD_classes_id = ['A', 'B1', 'B2', 'B4', 'B5', 'B6', 'G', ]

MSAD_classes = ['Normal', 'Assault', 'Explosion', 'Fighting', 'Fire', 'Object Falling', 'People Falling', 'Robbery',
    'Shooting', 'Traffic Accident', 'Vandalism', 'Water Incident']

def update_flat_scores_labels(flat_scores, flat_labels, video_scores, video_labels, without_labels, scores_json, video_name):
    class_text = get_video_category(scores_json, video_name)
    flat_scores['all'].extend(video_scores)
    if not without_labels:
        flat_labels['all'].extend(video_labels)
    if len(class_text) == 1:  # xd数据集中跳过多类的视频
        flat_scores[class_text[0]].extend(video_scores)
        if not without_labels:
            flat_labels[class_text[0]].extend(video_labels)
    if class_text[0] != 'Normal':
        flat_scores['Abnormal'].extend(video_scores)
        if not without_labels:
            flat_labels['Abnormal'].extend(video_labels)
    return flat_scores, flat_labels

def compute_metrics(flat_scores, flat_labels, normal_label, only_abnormal=False):
    """
    计算各类别的性能指标

    参数:
    flat_scores (dict): 各类别的预测分数字典
    flat_labels (dict): 各类别的真实标签字典
    normal_label (int): 正常样本的标签值，默认为0

    返回:
    dict: 包含各类别性能指标的字典
    """
    dataset_metric = {}

    # 计算指标
    if 'Normal' in flat_scores and only_abnormal:
        assert len(flat_labels['Normal']) == 0 # 说明本次测评没有正常类别，做此验证
        del flat_scores['Normal']

    for class_name in flat_scores:
        scores_class = np.array(flat_scores[class_name])
        labels_class = np.array(flat_labels[class_name])

        # 使用1表示该类别，0表示其他类别
        binary_labels = labels_class != normal_label
        # 计算ROC AUC
        fpr, tpr, _ = roc_curve(binary_labels, scores_class)
        roc_auc = round(auc(fpr, tpr), 4)

        # 计算PR AUC
        precision, recall, _ = precision_recall_curve(binary_labels, scores_class)
        pr_auc = round(auc(recall, precision), 4)

        # 计算正负样本的平均分数
        pos_mean = round(scores_class[binary_labels].mean(), 4)
        neg_mean = round(scores_class[~binary_labels].mean(), 4)

        print(f"Class: {class_name}, 1frames: {binary_labels.sum()}, 0frames: {(~binary_labels).sum()} ROC AUC:"
              f" {roc_auc}, PR AUC: {pr_auc}, ")

        dataset_metric[class_name] = {
            'pr_auc': pr_auc,
            'roc_auc': roc_auc,
            # '1_mean': pos_mean,
            # '0_mean': neg_mean,
        }

    return dataset_metric

def temporal_testing_annotations(temporal_annotation_file):
    annotations = {}

    with open(temporal_annotation_file) as annotations_f:
        for line in annotations_f:
            parts = line.strip().split()
            video_name = str(Path(parts[0]).stem)
            annotation_values = parts[2:]
            annotations[video_name] = annotation_values

    return annotations


def get_video_labels(video_record, annotations, normal_label):
    video_name = Path(video_record.path).name
    labels = []

    video_annotations = [x for x in annotations[video_name] if x != "-1"]

    # Separate start and stop indices
    start_indices = video_annotations[::2]
    stop_indices = video_annotations[1::2]

    for frame_index in range(video_record.num_frames):
        frame_label = normal_label

        # Check if the current frame index falls within any annotation range
        if len(video_record.label) == 1:
            for start_idx, end_idx, label in zip(
                    start_indices, stop_indices, video_record.label * len(start_indices)
            ):
                if int(start_idx) <= frame_index + video_record.start_frame <= int(end_idx):
                    frame_label = label
        else:
            video_labels = video_record.label

            # Pad video_labels if it's shorter than start_indices
            if len(video_labels) < len(start_indices):
                last_label = [video_record.label[-1]] * (len(start_indices) - len(video_labels))
                video_labels.extend(last_label)

            for start_idx, end_idx, label in zip(start_indices, stop_indices, video_labels):
                if int(start_idx) <= frame_index + video_record.start_frame <= int(end_idx):
                    frame_label = label

        labels.append(frame_label)

    return labels


def calculate_refine_scores(scores_dict, similarity_dict, similarity_type, topK, nn, dyn_ratio, tau, args):
    if isinstance(similarity_type, str): # 单种特征相似性
        scores_refine_dict = calculate_refine_type_scores(scores_dict, similarity_dict, similarity_type, topK, nn,
            dyn_ratio, tau, args)
    elif isinstance(similarity_type, list): # 多种特征相似性
        scores_refine_list = []
        for weight, i in similarity_type: # 得到每种refine类型的结果，并进行加权
            values_list = list(calculate_refine_type_scores(scores_dict, similarity_dict, i,
                topK,nn, dyn_ratio, tau, args).values())
            scores_refine_list.append([weight*j for j in  values_list])
        scores_refine_dict = {}
        for i, (k,v) in enumerate(scores_dict.items()):
            scores_refine_dict[k] = sum([refine_type[i] for refine_type in scores_refine_list]) # 取平均
    else:
        raise NotImplementedError
    return scores_refine_dict

def calculate_refine_type_scores(scores_dict, similarity_dict, similarity_type, topK, nn, dyn_ratio, tau, args):
    # 单种特征相似性
    scores_refine_dict = {}
    sum_scores = list(scores_dict.values())
    #scores_dict.keys() = dict_keys(['0, 577', '577, 580', '580, 884', '884, 1411'])
    clip_ = [list(map(lambda x: int(float(x)), clip.split(', '))) for clip in list(scores_dict.keys())]  # 视频片段
    clip_len = np.array([i[1]-i[0] for i in clip_])  # 视频片段长度
    clip_cen = np.array([int(i[1]+i[0]/2) for i in clip_])  # 视频片段中心位置

    if isinstance(sum_scores[0], list): # 带think info的得特殊处理
        sum_scores = [i[0] for i in sum_scores]
    sum_scores = np.array(sum_scores)
    if dyn_ratio!=None:  # nn取 max（比例， nn参数）
        nn = max(int(sum_scores.shape[0] * dyn_ratio), nn)
        # topK = nn
    M_sims = similarity_dict[similarity_type]
    for i, (k,v) in enumerate(scores_dict.items()):
        nn_M_sims = M_sims[i][max(0, i-nn//2):i+nn//2]  # nn邻域内的相似性分数
        nn_sum_scores = sum_scores[max(0, i-nn//2):i+nn//2]  # nn邻域内片段的原始异常打分
        max_idx = nn_M_sims.argsort()[::-1][:topK] # nn邻域内的相似性分数排序
        max_sum_scores = nn_sum_scores[max_idx]  # nn邻域内片段的原始异常打分取topK
        max_M_sims = nn_M_sims[max_idx]

        max_M_sims_stable = max_M_sims - np.max(max_M_sims)  # 增加稳定性处理
        weights = np.exp(max_M_sims_stable / tau) / np.sum(np.exp(max_M_sims_stable / tau))
        # weights = np.exp(max_M_sims/tau) / max(0.01, np.sum(np.exp(max_M_sims/tau)))
        # 新加一个功能，加权时根据视频片段的长度调整加权系数，长度越相似，系数越高
        scores_refine_dict[k] = np.sum(max_sum_scores * weights)
    return scores_refine_dict

def dur_refine(start, end, dur):
    if dur == 'entire' or dur == None:
        return start, end
    elif dur == 'second_half':
        return start + int((end-start)/2), end
    elif dur == 'first_half':
        return start, end - int((end-start)/2)
    else:
        raise NotImplementedError

def calculate_boost_scores(video_scores_dict, video_scores, ):
    video_scores = np.array(video_scores)
    for k,v in video_scores_dict.items():
        start, end = map(lambda x: int(float(x)), k.split(', '))
        if video_scores[start:end].mean()-v[0] > 0.15 and v[0]>0.5:
            clip_len = end-start
            video_scores[int(start+clip_len/4):int(end-clip_len/4)] = 1.
            print('boosting')
    # print('fdsf')
    return video_scores.tolist()

def get_flat_scores(scores_json, video_list):
    with open(scores_json) as f:
        data = json.load(f)
    result = []
    for video in video_list:
        video_name = Path(video.path).name
        scores = data["vid_score"][video_name + '.mp4']
        # 直接遍历已排序的区间键（假设JSON键已按帧顺序排列）
        for interval in sorted(scores.keys(), key=lambda x: int(x.split(',')[0])):
            start, end = map(int, interval.split(', '))
            result.extend([scores[interval][0]] * (end - start))
        # 补一个，不碍事
        result.extend([result[-1]])
    return result

def get_video_flat_scores(scores_dict):

    result = []
    # 直接遍历已排序的区间键（假设JSON键已按帧顺序排列）
    for interval in sorted(scores_dict.keys(), key=lambda x: int(x.split(',')[0])):
        start, end = map(int, interval.split(', '))
        result.extend([scores_dict[interval][0]] * (end - start))
    # 补一个，不碍事
    result.extend([result[-1]])
    return result

def args_to_dict(args, exclude_keys=None, max_depth=3):
    """
    将args对象转换为JSON可序列化的字典
    :param args: 参数对象（Namespace/class/dict）
    :param exclude_keys: 需要排除的键列表
    :param max_depth: 最大递归深度防止循环引用
    :return: 可序列化的字典
    """
    def convert_value(value, current_depth):
        if current_depth >= max_depth:
            return str(value)  # 防止无限递归

        # 处理常见不可序列化类型
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, Path):
            return str(value.resolve())
        if isinstance(value, (list, tuple, set)):
            return [convert_value(v, current_depth+1) for v in value]
        if isinstance(value, dict):
            return {k: convert_value(v, current_depth+1) for k, v in value.items()}

        # 处理对象属性
        try:
            return {
                k: convert_value(v, current_depth+1)
                for k, v in vars(value).items()
                if not k.startswith('_')  # 排除私有属性
            }
        except TypeError:
            return str(value)

    # 获取参数字典
    if isinstance(args, dict):
        args_dict = args
    else:
        args_dict = vars(args)  # 适用于argparse.Namespace或普通对象

    # 过滤排除项
    exclude = set(exclude_keys or [])
    filtered = {k: v for k, v in args_dict.items() if k not in exclude}

    # 递归转换值
    return convert_value(filtered, current_depth=0)
def initialize_score_dicts(scores_json):
    """
    初始化评分字典和标签字典

    参数:
    scores_json (str): 包含数据集标识的字符串（需包含'UCF'或'XD'）
    UCF_classes (list): UCF数据集的类别列表
    XD_classes (list): XD数据集的类别列表

    返回:
    tuple: 包含两个字典的元组 (flat_scores, flat_labels)

    异常:
    ValueError: 当scores_json不包含有效标识时抛出
    """
    # 初始化基础字典
    flat_scores = {'all': [], 'Abnormal': []}
    flat_labels = {'all': [], 'Abnormal': []}

    # 确定数据集类别
    if 'UCF' in scores_json.upper():
        dataset_classes = UCF_Crime_classes
    elif 'XD' in scores_json.upper():
        dataset_classes = XD_classes
        # 可以在此处添加XD的特殊处理逻辑
    elif 'MSAD' in scores_json.upper():
        dataset_classes = MSAD_classes
        # 可以在此处添加MSAD的特殊处理逻辑
    else:
        raise ValueError("Invalid scores_json identifier. Must contain 'UCF' or 'XD'")
    # 动态扩展字典
    for class_name in dataset_classes:
        # 跳过已存在的特殊类别
        if class_name in ('all', 'Abnormal'):
            continue

        # 初始化空列表（如果不存在）
        if class_name not in flat_scores:
            flat_scores[class_name] = []
        if class_name not in flat_labels:
            flat_labels[class_name] = []

    return flat_scores, flat_labels

def get_video_category(scores_json, video_name):
    """
    根据数据集类型和视频名称返回对应的类别字符串

    参数:
    dataset_type (str): 数据集类型，'UCF' 或 'XD'
    video_name (str): 视频文件名

    返回:
    str: 类别字符串，多个类别用逗号分隔，未找到时返回空字符串

    # 使用示例
    print(get_video_category('UCF', 'Abuse028_x264'))          # 输出: Abuse
    print(get_video_category('UCF', 'Normal_Videos_312_x264')) # 输出: Normal
    print(get_video_category('XD', 'Bad.Boys.1995__#01-33-51_01-34-37_label_B2-0-0'))  # 输出: Shooting
    print(get_video_category('XD', 'Bad.Boys.II.2003__#00-06-42_00-10-00_label_B2-G-0')) # 输出: Shooting,Explosion
    print(get_video_category('XD', 'Bad.Boys.II.2003__#01-11-16_01-14-00_label_A'))    # 输出: Normal
    """



    # 创建XD ID到类别的映射字典
    XD_id_mapping = dict(zip(XD_classes_id, XD_classes))



    #         dataset_classes = UCF_classes
    #
    # dataset_classes = XD_classes
    # # 可以在此处添加XD的特殊处理逻辑
    # else:
    # raise ValueError("Invalid scores_json identifier. Must contain 'UCF' or 'XD'")
    try:
        if 'UCF' in scores_json.upper():
            # 按类别名称长度降序排列（处理前缀匹配问题）
            sorted_classes = sorted(UCF_Crime_classes, key=lambda x: len(x), reverse=True)

            # 遍历所有可能的类别前缀
            for cls in sorted_classes:
                if video_name.startswith(cls):
                    return [cls]

        elif 'XD' in scores_json.upper():
            # 解析标签部分（格式：..._label_ID1-ID2-ID3）
            if '__' not in video_name:
                return ""

            # 提取标签部分
            label_segment = video_name.split('__')[-1].split('_label_')[-1]

            # 分割并过滤有效ID（去除含'0'的无效部分）
            raw_ids = [x for x in label_segment.split('-') if x != '0']

            # 转换为类别名称并去重
            categories = []
            seen = set()
            for vid in raw_ids:
                if vid in XD_id_mapping and XD_id_mapping[vid] not in seen:
                    categories.append(XD_id_mapping[vid])
                    seen.add(XD_id_mapping[vid])

            return categories
        elif 'MSAD' in scores_json.upper():
            # MSAD数据集的处理逻辑
            # 提取标签部分（格式：..._label_ID1-ID2-ID3）
            # 'Object_falling_3' 提取出 'Object Falling'， ‘Fire_11’ 提取出 'Fire'
            if 'normal' in video_name.lower():
                return ['Normal']
            segments = video_name.split('_')
            label_parts = [seg for seg in segments if not seg.isdigit()]
            categories = ' '.join(label_parts).replace('-', ' ').title()
            if categories not in MSAD_classes:
                raise ValueError(f"Category '{categories}' not found in MSAD classes.")
            return [categories]
        else:
            raise ValueError("Invalid dataset type. Must be 'UCF' or 'XD'")

    except Exception as e:
        print(f"Error processing {scores_json} video: {str(e)}")
        return ""

