import pickle5 as pickle
import numpy as np


def do_eval(gt_dict, pred_dict, head=None, threshold=0.05):
    # recall precision f1 for threshold 0.05(5%)
    tp_all = 0
    num_pos_all = 0
    num_det_all = 0

    for vid_id in list(gt_dict.keys()):

        # filter by avg_f1 score
        if gt_dict[vid_id]['f1_consis_avg'] < 0.3:
            continue

        if vid_id not in pred_dict.keys():
            # num_pos_all += len(gt_dict[vid_id]['substages_myframeidx'][0])
            continue

        # detected timestamps
        bdy_timestamps_det = pred_dict[vid_id]
        if head is not None:
            bdy_timestamps_det = bdy_timestamps_det[head]

        myfps = gt_dict[vid_id]['fps']
        ins_start = 0
        ins_end = gt_dict[vid_id]['num_frames'] - 1  # number of frames

        # remove detected boundary outside the action instance
        tmp = []
        for det in bdy_timestamps_det:
            tmpdet = det + ins_start
            if tmpdet >= (ins_start) and tmpdet <= (ins_end):
                tmp.append(tmpdet)
        bdy_timestamps_det = tmp
        if bdy_timestamps_det == []:
            num_pos_all += len(gt_dict[vid_id]['substages_myframeidx'][0])
            continue
        num_det = len(bdy_timestamps_det)
        num_det_all += num_det

        # compare bdy_timestamps_det vs. each rater's annotation, pick the one leading the best f1 score
        bdy_timestamps_list_gt_allraters = gt_dict[vid_id]['substages_myframeidx']
        f1_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))
        tp_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))
        num_pos_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))

        for ann_idx in range(len(bdy_timestamps_list_gt_allraters)):
            bdy_timestamps_list_gt = bdy_timestamps_list_gt_allraters[ann_idx]
            num_pos = len(bdy_timestamps_list_gt)
            tp = 0
            offset_arr = np.zeros((len(bdy_timestamps_list_gt), len(bdy_timestamps_det)))
            for ann1_idx in range(len(bdy_timestamps_list_gt)):
                for ann2_idx in range(len(bdy_timestamps_det)):
                    offset_arr[ann1_idx, ann2_idx] = abs(bdy_timestamps_list_gt[ann1_idx] - bdy_timestamps_det[ann2_idx])
            for ann1_idx in range(len(bdy_timestamps_list_gt)):
                if offset_arr.shape[1] == 0:
                    break
                min_idx = np.argmin(offset_arr[ann1_idx, :])
                if offset_arr[ann1_idx, min_idx] <= threshold * (ins_end - ins_start + 1):
                    tp += 1
                    offset_arr = np.delete(offset_arr, min_idx, 1)

            num_pos_tmplist[ann_idx] = num_pos
            fn = num_pos - tp
            fp = num_det - tp
            if num_pos == 0:
                rec = 1
            else:
                rec = tp / (tp + fn)
            if (tp + fp) == 0:
                prec = 0
            else:
                prec = tp / (tp + fp)
            if (rec + prec) == 0:
                f1 = 0
            else:
                f1 = 2 * rec * prec / (rec + prec)
            tp_tmplist[ann_idx] = tp
            f1_tmplist[ann_idx] = f1

        ann_best = np.argmax(f1_tmplist)
        tp_all += tp_tmplist[ann_best]
        num_pos_all += num_pos_tmplist[ann_best]

    fn_all = num_pos_all - tp_all
    fp_all = num_det_all - tp_all
    if num_pos_all == 0:
        rec = 1
    else:
        rec = tp_all / (tp_all + fn_all)
    if (tp_all + fp_all) == 0:
        prec = 0
    else:
        prec = tp_all / (tp_all + fp_all)
    if (rec + prec) == 0:
        f1 = 0
    else:
        f1 = 2 * rec * prec / (rec + prec)

    return f1, rec, prec


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
    exit_indices = []
    if len(bdy_indices) != 0:
        for internals in bdy_indices:
            center = round(np.mean(internals))
            bdy_indices_in_video.append(seq_indices[center])
            for inter in internals:
                exit_indices.append(seq_indices[inter])
    return bdy_indices_in_video, exit_indices


def early_exit_by_threshold(seq_scores, thh, thl, ignore=5, pad_ignore=2):
    seq_scores = np.array(seq_scores)
    positive_bdy_indices = []
    negtive_bdy_indices = []
    positive_indices = []
    negtive_indices = []

    T = len(seq_scores)

    for i in range(len(seq_scores)):

        if seq_scores[i] >= thh:
            positive_indices.append(i)
        elif seq_scores[i] < thh and len(positive_indices) != 0:
            positive_bdy_indices.append(positive_indices)
            positive_indices = []

        if seq_scores[i] <= thl:
            negtive_indices.append(i)
        elif seq_scores[i] > thl and len(negtive_indices) != 0:
            negtive_bdy_indices.append(negtive_indices)
            negtive_indices = []

        if i == len(seq_scores) - 1:
            if len(positive_indices) != 0:
                positive_bdy_indices.append(positive_indices)
            if len(negtive_indices) != 0:
                negtive_bdy_indices.append(negtive_indices)

    bdy_indices_in_video = []
    exit_idx = []
    psu_exit_idx = []
    minimal_ignore = ignore * 2 + 1

    # ensure exited clip is longer than $minimal_ignore frames.
    if len(positive_bdy_indices) != 0:
        for internals in positive_bdy_indices:
            center = round(np.mean(internals))
            bdy_indices_in_video.append(center)
            if len(internals) < minimal_ignore:
                exit_idx.append([i for i in range(max(0, center-ignore), min(T, center+ignore+1))])
                psu_exit_idx.append([i for i in range(max(0, center-ignore-pad_ignore), min(T, center+ignore+1+pad_ignore))])
            else:
                exit_idx.append([i for i in internals])
                psu_exit_idx.append([i for i in range(max(0, internals[0]-pad_ignore), min(T, internals[-1]+pad_ignore))])
    if len(negtive_bdy_indices) != 0:
        for internals in negtive_bdy_indices:
            center = round(np.mean(internals))
            if len(internals) > minimal_ignore:
                exit_idx.append([i for i in internals])
                psu_exit_idx.append([i for i in range(max(0, internals[0]-pad_ignore), min(T, internals[-1]+pad_ignore))])

    return bdy_indices_in_video, exit_idx, psu_exit_idx


def eval_f1(my_pred, gt_path, num_heads=1, threshold=0.5, return_pred_dict=False, rel_dis_thres=0.05):
    with open(gt_path, 'rb') as f:
        gt_dict = pickle.load(f, encoding='lartin1')

    pred_dict = dict()
    exit_dict = dict()
    for vid in my_pred:
        if vid in gt_dict:
            predictions = []
            exit_perdictions = []
            for head in range(num_heads):
                det_t, exit_t = get_idx_from_score_by_threshold(threshold=threshold,
                                                            seq_indices=my_pred[vid]['frame_idx'],
                                                            seq_scores=my_pred[vid]['scores'][head])
                assert np.all(np.array(det_t) >= 0)
                predictions.append(det_t)
                exit_perdictions.append(exit_t)
            pred_dict[vid] = predictions
            exit_dict[vid] = exit_perdictions

    if not isinstance(rel_dis_thres, list):
        rel_dis_thres = [rel_dis_thres]

    results = {}
    for thres in rel_dis_thres:
        res = []
        for head in range(num_heads):
            f1, rec, prec = do_eval(gt_dict, pred_dict, head, threshold=thres)
            res.append((f1, rec, prec))
        results[thres] = res

    if return_pred_dict:
        return results, pred_dict, exit_dict
        results = (results,) + (pred_dict, gt_dict)
    return results


def eval_f1_with_boundarys(my_pred, gt_path, rel_dis_thres=0.05):
    with open(gt_path, 'rb') as f:
        gt_dict = pickle.load(f, encoding='lartin1')

    pred_dict = dict()
    exit_percentages = np.zeros(2)
    idx = 0
    for vid in my_pred:
        if vid in gt_dict:
            predictions = [my_pred[vid]['frame_idx'][i] for i in my_pred[vid]['boundarys']]
            pred_dict[vid] = predictions
            exit_percentages += my_pred[vid]['exit_percentages']
            idx += 1
    exit_percentages /= idx

    results = {}
    for thres in rel_dis_thres:
        res = []
        f1, rec, prec = do_eval(gt_dict, pred_dict, threshold=thres)
        results[thres] = (f1, rec, prec)

    return results, exit_percentages
