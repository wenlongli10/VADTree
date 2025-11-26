import numpy as np
import torch, os
import torch.distributed as dist
from tabulate import tabulate
from utils.eval import eval_f1
import argparse, pickle


def get_result(args, dataset='GEBD'):

    with open(args.pred_file, 'rb') as f:
        model_pred_dict = pickle.load(f)
    print(f'Load results from {args.pred_file}.')

    num_heads = len([v for v in model_pred_dict.values()][0]['scores'])
    
    metrics = {}
    
    if dataset == 'GEBD':
        gt_path = os.path.join('data', f'k400_mr345_val_min_change_duration0.3.pkl')
    elif dataset == 'TAPOS':
        gt_path = os.path.join('data', f'tapos_gt_val.pkl')
    else:
        raise NotImplemented

    if args.all_thres:
        rel_dis_thres = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    else:
        rel_dis_thres = [0.05]

    results, pred_dict, gt_dict = eval_f1(model_pred_dict, gt_path,
                                          num_heads=num_heads,
                                          threshold=args.threshold,
                                          return_pred_dict=True,
                                          rel_dis_thres=rel_dis_thres)
        
    metrics_list = []

    for head in range(num_heads):

        list_rec = []
        list_prec = []
        list_f1 = []

        for th in rel_dis_thres:
            f1, rec, prec = results[th][head]
            list_rec.append(rec)
            list_prec.append(prec)
            list_f1.append(f1)

        headers = rel_dis_thres + ['Avg']

        avg_rec = np.mean(list_rec)
        avg_prec = np.mean(list_prec)
        avg_F1 = np.mean(list_f1)

        tabulate_data = [
            ['Recall'] + list_rec + [avg_rec],
            ['Precision'] + list_prec + [avg_prec],
            ['F1'] + list_f1 + [avg_F1],
        ]

        print(tabulate(tabulate_data, headers=headers, floatfmt='.4f'))

        f1, rec, prec = results[0.05][head]
        print('F1@0.05: {:.4f}, Rec: {:.4f}, Prec: {:.4f}'.format(f1, rec, prec))
        metrics = {}
        metrics['F1'] = f1
        metrics['Rec'] = rec
        metrics['Prec'] = prec
        
        metrics_list.append(metrics)

    return metrics_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_file", type=str)
    parser.add_argument("--all-thres", action='store_true', default=True, help='test using all thresholds [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]')
    parser.add_argument("-th", "--threshold", type=float, default=0.35)
    args = parser.parse_args()

    metrics_list = get_result(args, dataset='TAPOS')
