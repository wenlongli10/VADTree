import os
from os.path import join as ospj
import time
import subprocess
from tqdm import tqdm
from multiprocessing import Pool
import glob
import json


with open('data/TAPOS/tapos_annotation.json', 'r') as f:
    annos = json.load(f)


def par_job(command):
    subprocess.call(command, shell=True)

num_workers = 128

src_video_list = glob.glob('/data/TAPOS/videos/*.mp4')
src_video_list.sort()

if __name__ == "__main__":

    cmd_list = []
    for src_video in src_video_list:
        _, video_name = os.path.split(src_video)
        video_name = video_name[:-4]

        if video_name in annos.keys():
            for sub_name in annos[video_name]:
                # remove instances do not have boundaries
                if len(annos[video_name][sub_name]['substages']) > 2:
                    dst_video_name = '_'.join([video_name, sub_name])
                    start_time = float(sub_name.split('_')[1] + '.' + sub_name.split('_')[2])
                    end_time = float(sub_name.split('_')[3] + '.' + sub_name.split('_')[4])
                    abs_timestamps = annos[video_name][sub_name]['shot_timestamps']
                    start_time += abs_timestamps[0]
                    end_time += abs_timestamps[0]

                    split = annos[video_name][sub_name]['subset']
                    dst_img_dir, _ = os.path.split(src_video.replace('/videos', '/images-256/' + split))
                    dst_img_dir = os.path.join(dst_img_dir, dst_video_name)
                    if not os.path.exists(dst_img_dir):
                        os.makedirs(dst_img_dir)
                    cmd = 'ffmpeg -nostats -loglevel 0 -i \"{}\" -threads 1 -ss {} -to {} -vf scale=-1:256 -q:v 0 \"{}/frame%d.jpg\"' \
                        .format(src_video, start_time, end_time, dst_img_dir)
                    cmd_list.append(cmd)

    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(cmd_list)) as pbar:
            for _ in tqdm(pool.imap_unordered(par_job, cmd_list)):
                pbar.update()

    os.system("stty sane")
