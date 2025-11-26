import logging
import os
from metrics_fast import AverageMeter ,average_mAP,calculate_f1_score
import time
# from metrics_fast import average_mAP_visibility
from metrics_visibility_fast import average_mAP_visibility

from tqdm import tqdm
import torch
import numpy as np
import math
from tabulate import tabulate
from torch.nn.parallel import DistributedDataParallel
from distribute import synchronize, all_gather, is_main_process
from collections import defaultdict


def trainer(train_loader,
            test_loader,
            model,
            optimizer,
            scheduler,
            train_game_name,
            test_game_name,
            model_save_path,
            dis_thres=1,
            max_epochs=100):

    auto_cast = torch.cuda.amp.autocast
    loss_scaler = torch.cuda.amp.GradScaler()

    evaluate_test_epoch = 20
    save_checkpoint_epoch = 1
    best_f1 = 0.
    for epoch in range(max_epochs):
        best_model_path = os.path.join(model_save_path, "model_best.pth.tar")

        lr = scheduler.get_last_lr()[0]
        # train for one epoch
        train(
            train_loader,
            model,
            optimizer,
            epoch + 1,
            lr,
            auto_cast,
            loss_scaler)

        # if (epoch + 1) % evaluate_test_epoch == 0 or epoch == 0:

        # logging.info("Performance at epoch " + str(epoch+1))
        f1 = test(test_loader, model, dis_thres=dis_thres, game_name=test_game_name)
        # logging.info("Performance at epoch " + str(epoch+1) + " -> " + str(performance_test))

        model_state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
        state = {
            'epoch': epoch + 1,
            'model': model_state_dict,
            'best_f1': best_f1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        if (epoch + 1) % save_checkpoint_epoch == 0 or epoch == 0:
            torch.save(
                state,
                os.path.join(model_save_path, "model_epoch" + str(epoch + 1) + ".pth.tar"))

            # remember best prec@1 and save checkpoint
            is_better = f1 < best_f1
            best_f1 = max(f1, best_f1)

            # test the model
            if is_better:
                torch.save(state, best_model_path)

        scheduler.step()

    torch.save(state, os.path.join(model_save_path, "model_last.pth.tar"))
    return


def train(dataloader,
          model,
          optimizer,
          epoch,
          lr,
          auto_cast,
          loss_scaler):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    dataset_len = len(dataloader)
    iter_to_print = dataset_len // 10
    # with tqdm(enumerate(dataloader), total=len(dataloader), ncols=160) as t:
    for i, (feats, targets, masks, game_idxs, anchors) in enumerate(dataloader):

        data_time.update(time.time() - end)

        feats = feats.cuda()   # (b, t, c)
        targets = targets.cuda().float()   # (b, t)

        with auto_cast():
            # compute output
            loss = model(feats, targets, masks)

        # measure accuracy and record loss
        losses.update(loss.item(), feats.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_scaler.scale(loss).backward()
        loss_scaler.step(optimizer)
        loss_scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if is_main_process() and i % iter_to_print == 0:
            desc = f'Epoch {epoch:3d}: '
            desc += f'lr {lr:.2e} '
            desc += f'[{i / dataset_len * 100:.1f}]% '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            desc += f'Loss {losses.avg:.4f} '
            logging.info(desc)


@torch.no_grad()
def test(dataloader, model, dis_thres=1., prob_thres=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], game_name=None):
    data_time = AverageMeter()

    model.eval()

    end = time.time()
    # with tqdm(enumerate(dataloader), total=len(dataloader), ncols=120) as t:
    # for i, (feat_half1, feat_half2, label_change_half1, label_change_half2) in enumerate(dataloader):
    #     data_time.update(time.time() - end)

    #     feat_half1 = feat_half1.cuda()
    #     feat_half2 = feat_half2.cuda()

    #     # Compute the output
    #     scores_1 = model(feat_half1).squeeze(0).detach().cpu().numpy()
    #     scores_2 = model(feat_half2).squeeze(0).detach().cpu().numpy()

    #     score_labels.append([scores_1, label_change_half1.squeeze(0).numpy()])
    #     score_labels.append([scores_2, label_change_half2.squeeze(0).numpy()])

    output_dict = {name:[] for name in game_name}
    for i, (feats, targets, masks, game_idxs, anchors) in enumerate(dataloader):
        data_time.update(time.time() - end)
        feats = feats.cuda()
        scores = model(feats).detach().cpu()
        # per sample
        for s, t, m, g, a in zip(scores, targets, masks, game_idxs, anchors):
            output_dict[game_name[g]].append([s[m].numpy(), t[m].numpy(), a.numpy()])

    synchronize()
    merged_dict = defaultdict(list)
    data_list = all_gather(output_dict)
    for data in data_list:
        for key, value in data.items():
            merged_dict[key].extend(value)

    output_dict = dict(merged_dict)

    # a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown = average_mAP_visibility(onehot_labels, probs, model.framerate)
    fps = model.module.framerate if isinstance(model, DistributedDataParallel) else model.framerate
    results = eval(output_dict, dis_thres=dis_thres, prob_thres=prob_thres, fps=fps)

    list_rec = []
    list_prec = []
    list_f1 = []
    for th in prob_thres:
        f1, rec, prec = results[th]
        list_rec.append(rec)
        list_prec.append(prec)
        list_f1.append(f1)
    headers = prob_thres + ['Avg']
    avg_rec = np.mean(list_rec)
    avg_prec = np.mean(list_prec)
    avg_F1 = np.mean(list_f1)
    tabulate_data = [
        ['Recall'] + list_rec + [avg_rec],
        ['Precision'] + list_prec + [avg_prec],
        ['F1'] + list_f1 + [avg_F1],
    ]
    if is_main_process():
        logging.info(f'Distrance Threshold {dis_thres}s:\n' + tabulate(tabulate_data, headers=headers, floatfmt='.4f'))

    return avg_F1


def eval(output_dict, dis_thres=1., prob_thres=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], fps=5):

    results = {}
    for threshold in prob_thres:
        bdy_indices_all, gt_indices_all = [], []
        for game, output_list in output_dict.items():
            bdy_indices, gt_indices = [], []
            for score, gt, anchor in output_list:
                indice = np.arange(len(score)) + anchor
                bdy_indice = get_idx_from_score_by_threshold(indice, score, threshold)
                bdy_indices.extend(bdy_indice)
                gt_indice = np.where(gt == 1)[0] + anchor
                gt_indices.extend(gt_indice.tolist())
            bdy_indices_all.append(sorted(bdy_indices))
            gt_indices_all.append(sorted(gt_indices))

        results[threshold] = do_eval(gt_indices_all, bdy_indices_all, dis_thres=dis_thres*fps)

    return results


def get_idx_from_score_by_threshold(seq_indices=None, seq_scores=None, threshold=0.5):
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
            center = round(np.mean(internals))
            bdy_indices_in_video.append(seq_indices[center])
    return bdy_indices_in_video


def do_eval(gt_indices, bdy_indices, dis_thres):
    # recall precision f1 for threshold
    tp_all = 0
    num_pos_all = 0
    num_det_all = 0

    for gt_indice, bdy_indice in zip(gt_indices, bdy_indices):

        # detected timestamps
        bdy_timestamps_det = bdy_indice
        bdy_timestamps_list_gt = gt_indice

        if bdy_timestamps_det == []:
            num_pos_all += len(gt_indice)
            continue
        
        num_det = len(bdy_timestamps_det)
        num_det_all += num_det
        num_pos = len(bdy_timestamps_list_gt)
        num_pos_all += num_pos

        # compute precision and recall
        tp = 0
        offset_arr = np.zeros((len(bdy_timestamps_list_gt), len(bdy_timestamps_det)))
        # pairwise distance
        for ann1_idx in range(len(bdy_timestamps_list_gt)):
            for ann2_idx in range(len(bdy_timestamps_det)):
                offset_arr[ann1_idx, ann2_idx] = abs(bdy_timestamps_list_gt[ann1_idx] - bdy_timestamps_det[ann2_idx])
        # find tp that statisfies threshold
        for ann1_idx in range(len(bdy_timestamps_list_gt)):
            if offset_arr.shape[1] == 0:
                break
            min_idx = np.argmin(offset_arr[ann1_idx, :])
            if offset_arr[ann1_idx, min_idx] <= dis_thres:
                tp += 1
                offset_arr = np.delete(offset_arr, min_idx, 1)
        tp_all += tp

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
