import os
from os.path import join as ospj
import time
import subprocess
from tqdm import tqdm
from multiprocessing import Pool
import glob
import json


# split = 'train'

# anno_list = glob.glob(f'/data1/SoccerNet/{split}/*/*/*/Labels-cameras.json')

# print(len(anno_list))

# from SoccerNet.Downloader import getListGames
# a = getListGames(split, task="camera-changes")
# print(a)

# anno_path_lists = glob.glob('/data1/SoccerNet/test/*/*/*/*.json')
# anno_dict = {}

# for anno_path in anno_path_lists:
#     with open(anno_path, 'r') as file:
#         data = json.load(file)
#         name = data["UrlLocal"]
#         anno_dict[name] = data

# with open('/data1/SoccerNet/anno_test.json', 'w') as file:
#     json.dump(anno_dict, file)




# with open('/data1/SoccerNet/anno_train.json', 'r') as f:
#     annos = json.load(f)

def par_job(command):
    subprocess.call(command, shell=True)

num_workers = 128

src_video_list = glob.glob('/data1/SoccerNet/videos/val/*/*/*/*.mkv')
src_video_list.sort()

if __name__ == "__main__":

    cmd_list = []
    for src_video in src_video_list:
        src_video_name = src_video[:-4]
        split = src_video_name[-6]   # 1 or 2
        dst_img_dir, _ = os.path.split(src_video_name.replace('/videos', '/images-224'))
        dst_img_dir = os.path.join(dst_img_dir, split)
        if not os.path.exists(dst_img_dir):
            os.makedirs(dst_img_dir)
        cmd = 'ffmpeg -nostats -loglevel 0 -i \"{}\" -vf \"fps=2,scale=-1:224\" -q:v 0 \"{}/frame%d.jpg\"' \
            .format(src_video, dst_img_dir)
        cmd_list.append(cmd)
    
    # print(cmd_list)
    # assert 1==2
    
    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(cmd_list)) as pbar:
            for _ in tqdm(pool.imap_unordered(par_job, cmd_list)):
                pbar.update()

    os.system("stty sane")
