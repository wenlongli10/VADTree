import numpy as np
def all_node_ense(all_node_video_scores_dict,frames):
    """对一个视频内所有节点的异常分数进行平均，节点是二叉树的树形结构，返回融合后的分数列表。

    Args:
        all_node_scores_dict (dict): 所有节点的异常分数字典，格式为 {'0, 1400': [score1, des],'0, 546': [score2, des], '546,
        1204': [score3, des]...}。

    Returns:
        list: 融合后的异常分数list。
    """
    ense_scores_m = -np.ones((len(all_node_video_scores_dict), frames))
    ense_scores = []
    for idx, key in enumerate(all_node_video_scores_dict):
        score = all_node_video_scores_dict[key][0]
        s,e = map(int, key.split(', '))
        ense_scores_m[idx, s:e+1] = score  # 将score转换为一行中某区间的分数
    # 跳过-1区域，对矩阵非-1位置列向取平均
    for i in range(frames):
        valid_scores = ense_scores_m[:, i][ense_scores_m[:, i] != -1, ]
        if valid_scores.size > 0:
            ense_scores.append(np.mean(valid_scores))
        else:
            raise ValueError(f"frmae {i} 中没有有效分数，无法计算平均值。(all_node_video_scores_dict 覆盖性不足）")
    return ense_scores

def find_matched_segments(coarse_key, ense_scores):
    """根据粗粒度区间key和细粒度分数dict，返回所有被包含的细粒度区间key列表。"""
    c_start, c_end = map(int, coarse_key.split(', '))
    matched_segments = []
    for seg_key in ense_scores:
        seg_start, seg_end = map(int, seg_key.split(', '))
        if seg_start >= c_start and seg_end <= c_end:
            matched_segments.append(seg_key)
        elif seg_start > c_end:
            break
    return matched_segments
def dempster_combine(a, b):
    '''
    # # 示例数据
    # A = [0.6, 0.3, 0.5]  # 分类器A对三个样本的类别1概率
    # B = [0.4, 0.7, 0.1]  # 分类器B对三个样本的类别1概率
    #
    # # 逐个样本融合
    # fused = [dempster_combine(a, b) for a, b in zip(A, B)]
    #
    # print("融合后类别1的置信度:", [round(x, 4) for x in fused])

    :param a:
    :param b:
    :return:
    '''
    """融合两个分类器对类别1的置信度，返回融合后的置信度（Belief for C1）。"""
    # 基本概率分配：m1({C1})=a, m1(Θ)=1-a；m2({C1})=b, m2(Θ)=1-b
    # 应用 Dempster 规则计算融合后的 m({C1})
    K = 0.0  # 冲突系数（此场景下无冲突，K=0）
    m_c1 = (a * b) + (a * (1 - b)) + ((1 - a) * b)
    m_c1_normalized = m_c1 / (1 - K)  # 归一化
    return m_c1_normalized


def apply_normalization(data, method):
    if method == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == 'softmax':
        # 数值稳定型softmax
        data = np.array(data)
        exp_data = np.exp(data - np.max(data))  # 防溢出处理
        return exp_data / exp_data.sum()
    else:
        raise ValueError(f"不支持的归一化方法: {method}")

def compute_normalized_variances(coarse_intervals, ense_scores, ense_vid_feats=None, ense_text_feats=None,
        normalization='minmax'):
    """计算并返回标准化后的三种方差

    Args:
        coarse_intervals (dict): 粗粒度区间字典，格式 {'start, end': [...]}
        ense_scores (dict): 细粒度异常分数，格式 {'start, end': [score]}
        ense_vid_feats (list): 视频特征列表，顺序与ense_scores的keys()一致
        ense_text_feats (list): 文本特征列表，顺序与ense_scores的keys()一致

    Returns:
        dict: 包含标准化方差的结果字典，格式 {
            'coarse_interval': {
                'score_var': 原始方差,
                'vid_var': 原始方差,
                'text_var': 原始方差,
                'norm_score': 标准化值,
                'norm_vid': 标准化值,
                'norm_text': 标准化值
            },
            ...
        }
    """
    # 预处理：将特征列表转为字典
    if ense_vid_feats != None:
        vid_feat_dict = {}
        text_feat_dict = {}
        for idx, seg_key in enumerate(ense_scores.keys()):
            vid_feat_dict[seg_key] = ense_vid_feats[idx]
            text_feat_dict[seg_key] = ense_text_feats[idx]

    # 阶段1：计算所有原始方差
    variance_data = {}
    for coarse_key in coarse_intervals:
        # 解析粗粒度区间
        c_start, c_end = map(int, coarse_key.split(', '))

        # 寻找对应的细粒度区间
        matched_segments = []
        for seg_key in ense_scores:
            seg_start, seg_end = map(int, seg_key.split(', '))

            # 区间完全包含时记录
            if seg_start >= c_start and seg_end <= c_end:
                matched_segments.append(seg_key)
            # 超出当前粗区间时提前终止
            elif seg_start > c_end:
                break

        # 计算异常分数方差
        fine_scores = []
        for seg in matched_segments:
            # 处理分数格式 (兼容列表或标量)
            score = ense_scores[seg]
            fine_scores.append(score[0] if isinstance(score, list) else score)

        # print('coarse:', coarse_intervals[coarse_key][0], 'fine:',fine_scores)

        score_var = np.var(fine_scores) if fine_scores else 0.0

        # 计算zscore
        coarse_score = coarse_intervals[coarse_key][0]
        coarse_zscore = abs(coarse_score-np.mean(fine_scores))/(np.std(fine_scores) + 0.001)

        if ense_vid_feats != None:
            # 计算视频特征方差

            vid_features = []
            for seg in matched_segments:
                vid_features.append(vid_feat_dict[seg])
            vid_matrix = np.concatenate(vid_features, axis=0) if vid_features else np.zeros((0, 1024))
            vid_var = np.mean(np.var(vid_matrix, axis=0)) if vid_matrix.size > 0 else 0.0

            # 计算文本特征方差
            text_features = []
            for seg in matched_segments:
                text_features.append(text_feat_dict[seg])
            text_matrix = np.concatenate(text_features, axis=0) if text_features else np.zeros((0, 512))
            text_var = np.mean(np.var(text_matrix, axis=0)) if text_matrix.size > 0 else 0.0

            variance_data[coarse_key] = (score_var, vid_var, text_var, coarse_zscore)
        else:
            variance_data[coarse_key] = (score_var, 0., 0., coarse_zscore)

    # 阶段2：全局归一化

    # 对每个维度分别应用归一化
    score_values = [v[0] for v in variance_data.values()]
    vid_values = [v[1] for v in variance_data.values()]
    text_values = [v[2] for v in variance_data.values()]
    coarse_zscore_values = [v[3] for v in variance_data.values()]

    norm_scores = apply_normalization(score_values, normalization)
    norm_vid = apply_normalization(vid_values, normalization)
    norm_text = apply_normalization(text_values, normalization)
    norm_zscore = apply_normalization(coarse_zscore_values, normalization)
    # print('coarse_zscore_values norm_zscore:',coarse_zscore_values, norm_zscore)
    # 构建结果字典
    result = {}
    for idx, key in enumerate(variance_data):
        result[key] = {
            'raw_score': float(variance_data[key][0]),
            'raw_vid': float(variance_data[key][1]),
            'raw_text': float(variance_data[key][2]),
            'zscore': float(variance_data[key][3]),

            'norm_score': float(norm_scores[idx]),
            'norm_vid': float(norm_vid[idx]),
            'norm_text': float(norm_text[idx]),
            'norm_zscore': float(norm_zscore[idx]),

            'normalization': normalization,
        }
    return result
# 使用示例
# video_variances = calculate_variances(
#     video_scores_dict, ense_video_scores_dict, ense_vid_feat, ense_vid_text_feat
# )
#
# # 打印结果样例
# for key, vals in video_variances.items():
#     print(f"区间 {key}:")
#     print(f"异常分数方差: {vals['score_variance']:.6f}")
#     print(f"视频特征方差: {vals['video_feat_variance']:.6f}")
#     print(f"文本特征方差: {vals['text_feat_variance']:.6f}\n")