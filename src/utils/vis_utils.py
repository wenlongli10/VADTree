import os
import textwrap
from decord import VideoReader, cpu

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
# plt.style.use('science')
import numpy as np
from itertools import chain
# import scienceplots
from matplotlib import font_manager
import matplotlib.gridspec as gridspec

font_path = os.path.join(os.path.dirname(__file__),'fonts/TIMES.TTF')
# 使用 font_manager 加载字体
prop = font_manager.FontProperties(fname=font_path)

def find_closest_key_value(d, frame_idx):
    sorted_items = sorted(
        (int(key), dict(value)) for key, value in d.items() if int(key) <= frame_idx
    )
    return sorted_items[-1] if sorted_items else (None, None)
def find_frame_caption(video_captions, i):
    for k, v in video_captions.items():
        start, end = map(lambda x: int(float(x)), k.split(', '))
        if start<=i and  i<=end:
            if type(v) is list:
                return v[-1]
            else:
                return v
def visualize_video_paper(
        video_name,
        video_labels,
        video_scores,
        video_captions,
        video_path,
        video_fps,
        save_path,
        normal_label,
        imagefile_template,
        optimal_threshold=0.5,
        font_size=18,
        save_video=True,
        video_metric='',
        scenes=None,
        split_pred=None,
):

    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 2, width_ratios=[0.4, 0.6], height_ratios=[0.6, 0.4])

    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, :])

    video_writer = None

    # 画预测结果
    x = np.arange(len(video_scores))
    ax3.plot(x, video_scores, color="#4e79a7", linewidth=2)
    ymin, ymax = 0, 1.01
    xmin, xmax = 0, len(video_scores)
    ax3.set_xlim([xmin, xmax])
    ax3.set_ylim([ymin, ymax])
    title = video_name

    # if not plot_split and split_pred is None:
    #     raise ValueError(f'plot_split {plot_split} and split_pred is None')
    if split_pred is not None:
        # 画split pred
        if "pred" in split_pred:
            split_pred_p = np.array(split_pred['pred'])

            x = np.linspace(0, split_pred['frames']-1, len(split_pred_p), dtype=int)
            ax3.scatter(x, split_pred_p, color="red", s=15,label='Points')

        # split_pred_p_pow = np.power(split_pred_p, 0.3)
        # ax3.scatter(x, split_pred_p_pow, color="red", s=30,label='Points')
        # ax3.scatter(x, split_pred['pred'], color="red", s=30,label='Points')
        # ax3.scatter(list(chain(*split_pred['scenes'])), [1.]*len(split_pred['scenes']*2), color="black", s=30, marker='^',label='Points')
    # 画split scenes
    if scenes is not None:
        split_scenes = list(set(scenes))
        split_scenes.sort()
        ax3.scatter(split_scenes, [1 for i in split_scenes], color="green", s=30,label='Points')

    start_idx = None
    text_flag = 0  # 错位写字的flag

    # 画groundtruth
    for frame_idx, label in enumerate(video_labels):
        if label != normal_label and start_idx is None:
            start_idx = frame_idx
        elif label == normal_label and start_idx is not None:
            rect = plt.Rectangle(
                (start_idx, ymin), frame_idx - start_idx, ymax - ymin, color="#e15759", alpha=0.5
            )
            ax3.add_patch(rect)
            # ax3.text(start_idx, ymin + 0.07*text_flag, f"{start_idx}-{frame_idx}", fontproperties=prop,fontsize=10, color='green',
            #          verticalalignment='bottom', horizontalalignment='left')
            text_flag = 1 if text_flag == 0 else 0
            start_idx = None

    if start_idx is not None:
        rect = plt.Rectangle(
            (start_idx, ymin),
            len(video_labels) - start_idx,
            ymax - ymin,
            color="#e15759",
            alpha=0.5,
        )
        ax3.add_patch(rect)
        # 画groundtruth 起止帧
        # ax3.text(start_idx, ymin, f"{start_idx}-{frame_idx}", fontproperties=prop,fontsize=15, color='green',
        #          verticalalignment='bottom', horizontalalignment='left')

    ax3.text(0.02, 0.90, f'{title}' ,fontproperties=prop,fontsize=14, transform=ax3.transAxes)
    # ax3.text(0.02, 0.90, f'{title}   frames:{len(video_scores)}   {video_metric}', fontproperties=prop,fontsize=14, transform=ax3.transAxes)

    for y_value in [0.25, 0.5, 0.75, 1]:
        ax3.axhline(y=y_value, color="grey", linestyle="--", linewidth=0.8)

    for x_value in range(0,len(video_scores), int(video_fps)):
        ax3.axvline(x=x_value, color="grey", linestyle="--", linewidth=0.8)

    # 网格
    ax3.set_yticks([0.25, 0.5, 0.75])

    # frame_tick = list(range(0,len(video_scores), int(video_fps) * max(round(len(video_scores)/1000), 1)))
    frame_tick = list(range(0,int(len(video_scores)*0.9), 1))
    frame_sec_text = [f"{i}\n{i/video_fps:.0f}" for i in frame_tick]
    frame_sec_text = [f"{i}\n{i/video_fps:.0f}" for i in frame_tick]

    # 计算近似等距的采样索引（包含首尾）
    n_total = len(frame_tick)
    indices = np.linspace(0, n_total-1, num=10, dtype=int)  # 生成10个等距索引
    # 根据索引采样
    sampled_ticks = [frame_tick[i] for i in indices]
    sampled_texts = [frame_sec_text[i] for i in indices]
    print(sampled_ticks)
    ax3.set_xticks(sampled_ticks)
    ax3.set_xticklabels(sampled_texts)
    # plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # 限制刻度数量
    # ax3.set_xticks(frame_tick)
    # ax3.set_xticklabels(frame_sec_text)

    ax3.tick_params(axis="y", labelsize=font_size-4)
    ax3.tick_params(axis="x", labelsize=font_size-4)

    # plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # 限制刻度数量
    ax3.set_ylabel("Anomaly score", fontproperties=prop,fontsize=font_size)
    # ax3.set_xlabel("Frame/Sencond", fontproperties=prop,fontsize=font_size)
    ax3.set_xlabel("Frame/Sencond", fontproperties=prop,fontsize=font_size)
    # ax3.set_xlabel("Frame/Sencond number", fontproperties=prop,fontsize=12)
    previous_line = None

    ## 保存子图
    bbox = ax3.get_tightbbox(fig.canvas.get_renderer()).expanded(1.02, 1.1)
    extent = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(str(save_path).replace('.mp4', '.png'), bbox_inches=extent,
        # dpi=150,
        transparent=True,  # 可选：透明背景
        dpi=100
    )


def visualize_video(
        video_name,
        video_labels,
        video_scores,
        video_captions,
        video_root,
        video_fps,
        save_path,
        normal_label,
        imagefile_template,
        optimal_threshold=0.5,
        font_size=25,
        save_video=True,
        video_metric='',
        scenes=None,
        split_pred=None,
):
    if save_video:
        fig = plt.figure(figsize=(20, 12))
    else:
        fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[0.4, 0.6], height_ratios=[0.6, 0.4])

    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, :])

    video_writer = None

    # 画预测结果
    x = np.arange(len(video_scores))
    ax3.plot(x, video_scores, color="#4e79a7", linewidth=2)
    ymin, ymax = 0, 1.01
    xmin, xmax = 0, len(video_scores)
    ax3.set_xlim([xmin, xmax])
    ax3.set_ylim([ymin, ymax])
    title = video_name

    # if not plot_split and split_pred is None:
    #     raise ValueError(f'plot_split {plot_split} and split_pred is None')
    if split_pred is not None:
        # 画split pred
        if "pred" in split_pred:
            split_pred_p = np.array(split_pred['pred'])

            x = np.linspace(0, split_pred['frames']-1, len(split_pred_p), dtype=int)
            ax3.scatter(x, split_pred_p, color="red", s=15,label='Points')

        # split_pred_p_pow = np.power(split_pred_p, 0.3)
        # ax3.scatter(x, split_pred_p_pow, color="red", s=30,label='Points')
        # ax3.scatter(x, split_pred['pred'], color="red", s=30,label='Points')
        # ax3.scatter(list(chain(*split_pred['scenes'])), [1.]*len(split_pred['scenes']*2), color="black", s=30, marker='^',label='Points')
    # 画split scenes
    if scenes is not None:
        split_scenes = list(set(scenes))
        split_scenes.sort()
        ax3.scatter(split_scenes, [1 for i in split_scenes], color="green", s=30,label='Points')

    start_idx = None
    text_flag = 0  # 错位写字的flag

    # 画groundtruth
    for frame_idx, label in enumerate(video_labels):
        if label != normal_label and start_idx is None:
            start_idx = frame_idx
        elif label == normal_label and start_idx is not None:
            rect = plt.Rectangle(
                (start_idx, ymin), frame_idx - start_idx, ymax - ymin, color="#e15759", alpha=0.5
            )
            ax3.add_patch(rect)
            # ax3.text(start_idx, ymin + 0.07*text_flag, f"{start_idx}-{frame_idx}", fontproperties=prop,fontsize=10, color='green',
            #          verticalalignment='bottom', horizontalalignment='left')
            text_flag = 1 if text_flag == 0 else 0
            start_idx = None

    if start_idx is not None:
        rect = plt.Rectangle(
            (start_idx, ymin),
            len(video_labels) - start_idx,
            ymax - ymin,
            color="#e15759",
            alpha=0.5,
        )
        ax3.add_patch(rect)
        # 画groundtruth 起止帧
        # ax3.text(start_idx, ymin, f"{start_idx}-{frame_idx}", fontproperties=prop,fontsize=15, color='green',
        #          verticalalignment='bottom', horizontalalignment='left')

    ax3.text(0.02, 0.90, f'{title}' ,fontproperties=prop,fontsize=font_size, transform=ax3.transAxes)
    # ax3.text(0.02, 0.90, f'{title}   frames:{len(video_scores)}   {video_metric}', fontproperties=prop,fontsize=14, transform=ax3.transAxes)

    for y_value in [0.25, 0.5, 0.75, 1]:
        ax3.axhline(y=y_value, color="grey", linestyle="--", linewidth=0.8)

    for x_value in range(0,len(video_scores), int(video_fps)):
        ax3.axvline(x=x_value, color="grey", linestyle="--", linewidth=0.8)

    # 网格
    ax3.set_yticks([0.25, 0.5, 0.75])

    # frame_tick = list(range(0,len(video_scores), int(video_fps) * max(round(len(video_scores)/1000), 1)))
    frame_tick = list(range(0,int(len(video_scores)*0.97), 1))
    # frame_sec_text = [f"{i}\n{i/video_fps:.0f}" for i in frame_tick]
    frame_sec_text = [f"{i}/{i/video_fps:.0f}" for i in frame_tick]

    # 计算近似等距的采样索引（包含首尾）
    n_total = len(frame_tick)
    indices = np.linspace(0, n_total-1, num=7, dtype=int)  # 生成10个等距索引
    # 根据索引采样
    sampled_ticks = [frame_tick[i] for i in indices]
    sampled_texts = [frame_sec_text[i] for i in indices]
    print(sampled_ticks)
    ax3.set_xticks(sampled_ticks)
    ax3.set_xticklabels(sampled_texts, fontproperties=prop)
    # plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # 限制刻度数量
    # ax3.set_xticks(frame_tick)
    # ax3.set_xticklabels(frame_sec_text)

    ax3.tick_params(axis="y", labelsize=font_size-4)
    ax3.tick_params(axis="x", labelsize=font_size-4)

    # plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # 限制刻度数量
    ax3.set_ylabel("Anomaly score", fontproperties=prop,fontsize=font_size)
    # ax3.set_xlabel("Frame/Sencond", fontproperties=prop,fontsize=font_size)
    ax3.set_xlabel("Frame/Sencond", fontproperties=prop,fontsize=font_size)
    # ax3.set_xlabel("Frame/Sencond number", fontproperties=prop,fontsize=12)
    previous_line = None

    if save_video== False:
        ## 保存子图
        bbox = ax3.get_tightbbox(fig.canvas.get_renderer()).expanded(1.02, 1.1)
        extent = bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(str(save_path).replace('.mp4', '.png'), bbox_inches=extent,
            # dpi=150,
            transparent=True,  # 可选：透明背景
            dpi=100
        )
        plt.close()
        # cv2.destroyAllWindows()
        return
    # start save video vis
    vid_path = os.path.join(video_root, video_name + '.mp4')
    vr = VideoReader(vid_path, ctx=cpu(1), num_threads=1)

    # ax2.set_title("Temporal summary", fontproperties=prop,fontsize=font_size)
    for ax in [ax2]:
        ax.invert_xaxis()  # 或者更简单的是不要调用 invert_xaxis()
    for i, score in enumerate(video_scores):
        if i % 5 != 0:
            continue
        ax1.set_title(f"Video frame: {i}  score: {score:.2f}", fontproperties=prop,fontsize=font_size)


        # img_name = imagefile_template.format(i)
        # img_path = os.path.join(video_path, img_name)
        # img = cv2.imread(img_path)

        img = vr.get_batch([i]).asnumpy()[0]

        if video_labels:
            box_color = (255, 0, 0) if score < optimal_threshold else (0, 0, 255)
            cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), box_color, 5)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax1.imshow(img)
        ax1.axis("off")

        # Display captions in a box on ax2
        # clip_frame_idx, clip_caption = find_closest_key_value(video_captions[0], i)
        # frame_caption = clip_caption.get(str(0), "")

        frame_caption = find_frame_caption(video_captions[0], i)
        frame_caption = 'COARSE: ' + frame_caption
        wrapped_caption = textwrap.fill(frame_caption, width=100)  # Adjust the width as needed

        ax2.text(
            0.5,
            0.5,
            wrapped_caption,
            fontproperties=prop,fontsize=18,
            verticalalignment="center",
            horizontalalignment="center",
            bbox=dict(
                facecolor="white",
                alpha=0.7,
                boxstyle="round",
                pad=0.5,
                edgecolor="black",
                linewidth=2,
            ),
            transform=ax2.transAxes,
            wrap=True,
        )
        ax2.axis("off")

        # Update or create the axvline
        if previous_line is not None:
            # Clear previous axvline
            previous_line.remove()

        axvline = ax3.axvline(x=i, color="red")

    # bbox = ax3.get_tightbbox(fig.canvas.get_renderer()).expanded(1.02, 1.1)
    # extent = bbox.transformed(fig.dpi_scale_trans.inverted())
    #     ax3.invert_xaxis()

        # fig.tight_layout()

        if video_writer is None:
            fig_size = fig.get_size_inches() * fig.dpi
            video_width, video_height = int(fig_size[0]), int(fig_size[1])
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                str(save_path), fourcc, video_fps, (video_width, video_height)
            )

        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        img = cv2.flip(img, 1)
        video_writer.write(img)

        ax1.cla()
        ax2.cla()

        # Update previous_line
        previous_line = axvline

    plt.close()
    video_writer.release()
    cv2.destroyAllWindows()



