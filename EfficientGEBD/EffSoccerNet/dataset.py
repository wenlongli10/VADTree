from torch.utils.data import Dataset

import numpy as np
# import pandas as pd
import os
import time


from tqdm import tqdm
# import utils

import torch

import logging
import json

from SoccerNet.Downloader import getListGames
from SoccerNet.Downloader import SoccerNetDownloader
from config.classes import EVENT_DICTIONARY_V1, EVENT_DICTIONARY_V2, K_V1, K_V2,Camera_Change_DICTIONARY,Camera_Type_DICTIONARY

from preprocessing import oneHotToAlllabels, getTimestampTargets, getChunks, getChunks_anchors



class SoccerNetClips(Dataset):
    def __init__(self, path, features="ResNET_PCA512.npy", split="train", framerate=5, chunk_size=60, receptive_field=6):
        self.path = path
        self.split = split
        self.listGames = getListGames(split, task="camera-changes")
        self.features = features
        self.chunk_size = chunk_size * framerate
        self.receptive_field = receptive_field * framerate
        self.framerate = framerate
        self.dict_change = Camera_Change_DICTIONARY
        self.labels = "Labels-cameras.json"

        # logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)
        downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[split], verbose=False, task="camera-changes", randomized=True)

        # logging.info("Pre-compute clips")

        self.game_name, self.game_feats, self.game_change_labels, self.game_anchors = [], [], [], []
        game_counter = 0
        for game in tqdm(self.listGames):
            feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features))
            feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features))

            # Load labels
            labels = json.load(open(os.path.join(path, game, self.labels)))

            nf1, nf2 = feat_half1.shape[0], feat_half2.shape[0]
            label_change_half1 = np.zeros(nf1)
            label_change_half2 = np.zeros(nf2)

            for annotation in labels["annotations"]:
                time = annotation["gameTime"]
                camera_change=annotation["change_type"]
                half = int(time[0])

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                framerate=self.framerate
                frame = framerate * ( seconds + 60 * minutes )

                # Onehot for camera change
                if camera_change in self.dict_change:
                
                    label_change = self.dict_change[camera_change]

                    if half == 1:
                        frame = min(frame, feat_half1.shape[0]-1)
                        label_change_half1[frame] = 1

                    if half == 2:
                        frame = min(frame, feat_half2.shape[0]-1)
                        label_change_half2[frame] = 1

            self.game_name.append(f"{game}-1")
            self.game_name.append(f"{game}-2")
            self.game_feats.append(feat_half1)
            self.game_feats.append(feat_half2)
            self.game_change_labels.append(label_change_half1)
            self.game_change_labels.append(label_change_half2)

            for anchor in range(0, nf1, self.chunk_size):
                self.game_anchors.append([game_counter, anchor])
            game_counter += 1
            for anchor in range(0, nf2, self.chunk_size):
                self.game_anchors.append([game_counter, anchor])
            game_counter += 1


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            feat_half1 (np.array): features for the 1st half.
            label_half1 (np.array): labels (one-hot) for the 1st half.
        """

        # Retrieve the game index and the anchor
        game_index = self.game_anchors[index][0]
        anchor = self.game_anchors[index][1]

        mask = np.ones(self.chunk_size, dtype=bool)
        # Extract the clips
        if anchor+self.chunk_size >= self.game_feats[game_index].shape[0]:
            # repeated padding
            clip_feat = self.game_feats[game_index][anchor:]
            last_feat = clip_feat[-1]
            num_repeats = self.chunk_size - clip_feat.shape[0]
            repeated_feat = np.tile(last_feat, (num_repeats, 1))
            clip_feat = np.concatenate((clip_feat, repeated_feat), axis=0)

            clip_change_labels = self.game_change_labels[game_index][anchor:]
            repeated_feat = np.zeros(num_repeats)
            clip_change_labels = np.concatenate((clip_change_labels, repeated_feat), axis=0)

            mask[clip_feat.shape[0]:] = False

        else:
            clip_feat = self.game_feats[game_index][anchor:anchor+self.chunk_size]
            clip_change_labels = self.game_change_labels[game_index][anchor:anchor+self.chunk_size]

        return torch.from_numpy(clip_feat), torch.from_numpy(clip_change_labels), torch.from_numpy(mask), torch.tensor([game_index]), torch.tensor([anchor])

    def __len__(self):
        return len(self.game_anchors)


# class SoccerNetClips(Dataset):
#     def __init__(self, path, features="ResNET_PCA512.npy", split="train", framerate=5, chunk_size=60, receptive_field=6):
#         self.path = path
#         self.listGames = getListGames(split, task="camera-changes")
#         self.features = features
#         self.chunk_size = chunk_size* framerate
#         self.receptive_field = receptive_field* framerate
#         self.framerate = framerate

#         self.dict_change = Camera_Change_DICTIONARY
#         self.labels = "Labels-cameras.json"

#         # logging.info("Checking/Download features and labels locally")
#         downloader = SoccerNetDownloader(path)
#         downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[split], verbose=False, task="camera-changes", randomized=True)

#         # logging.info("Pre-compute clips")

#         self.game_feats = list()
#         self.game_labels = list()
#         self.game_change_labels = list()
#         self.game_anchors = list()

#         game_counter = 0
#         for game in tqdm(self.listGames):
#             feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features))
#             feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features))

#             # Load labels
#             labels = json.load(open(os.path.join(path, game, self.labels)))

#             nf1, nf2 = feat_half1.shape[0], feat_half2.shape[0]
#             label_change_half1 = np.zeros(nf1)
#             label_change_half2 = np.zeros(nf2)

#             for annotation in labels["annotations"]:
#                 time = annotation["gameTime"]
#                 camera_change = annotation["change_type"]
#                 half = int(time[0])

#                 minutes = int(time[-5:-3])
#                 seconds = int(time[-2::])
#                 framerate = self.framerate
#                 frame = framerate * ( seconds + 60 * minutes )

#                 # Onehot for camera change
#                 if camera_change in self.dict_change:
                
#                     label_change = self.dict_change[camera_change]

#                     if half == 1:
#                         frame = min(frame, feat_half1.shape[0]-1)
#                         label_change_half1[frame] = 1

#                     if half == 2:
#                         frame = min(frame, feat_half2.shape[0]-1)
#                         label_change_half2[frame] = 1

#             anchors_half1 = getChunks_anchors(label_change_half1, game_counter)
#             game_counter += 1
#             anchors_half2 = getChunks_anchors(label_change_half2, game_counter)
#             game_counter += 1

#             self.game_feats.append(feat_half1)
#             self.game_feats.append(feat_half2)
#             self.game_change_labels.append(label_change_half1)
#             self.game_change_labels.append(label_change_half2)

#             for anchor in anchors_half1:
#                 self.game_anchors.append(anchor)
#             for anchor in anchors_half2:
#                 self.game_anchors.append(anchor)


#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#         Returns:
#             feat_half1 (np.array): features for the 1st half.
#             label_half1 (np.array): labels (one-hot) for the 1st half.
#         """

#         # Retrieve the game index and the anchor
#         game_index = self.game_anchors[index][0]
#         anchor = self.game_anchors[index][1]

#         # Compute the shift
#         shift = np.random.randint(-self.chunk_size+self.receptive_field, -self.receptive_field)
#         start = anchor + shift
#         if start < 0:
#             start = 0
#         if start+self.chunk_size >= self.game_feats[game_index].shape[0]:
#             start = self.game_feats[game_index].shape[0]-self.chunk_size-1

#         # Extract the clips
#         clip_feat = self.game_feats[game_index][start:start+self.chunk_size]
#         clip_change_labels = self.game_change_labels[game_index][start:start+self.chunk_size]

#         return torch.from_numpy(clip_feat), torch.from_numpy(clip_change_labels)

#     def __len__(self):
#         return len(self.game_anchors)


class SoccerNetClipsTesting(Dataset):
    def __init__(self, path, features="ResNET_PCA512.npy", split="train", framerate=5, chunk_size=60, receptive_field=6):
        self.path = path
        self.listGames = getListGames(split, task="camera-changes")
        self.features = features
        self.chunk_size = chunk_size * framerate
        self.receptive_field = receptive_field * framerate
        self.framerate = framerate

        self.dict_change = Camera_Change_DICTIONARY
        self.labels="Labels-cameras.json"

        # logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)
        downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[split], verbose=False, task="camera-changes", randomized=True)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            feat_half1 (np.array): features for the 1st half.
            feat_half2 (np.array): features for the 2nd half.
            label_half1 (np.array): labels (one-hot) for the 1st half.
            label_half2 (np.array): labels (one-hot) for the 2nd half.
        """
        # Load features
        feat_half1 = np.load(os.path.join(self.path, self.listGames[index], "1_" + self.features))
        feat_half2 = np.load(os.path.join(self.path, self.listGames[index], "2_" + self.features))

        # Load labels
        labels = json.load(open(os.path.join(self.path, self.listGames[index], self.labels)))
        
        nf1, nf2 = feat_half1.shape[0], feat_half2.shape[0]
        label_change_half1 = np.zeros(nf1)
        label_change_half2 = np.zeros(nf2)

        for annotation in labels["annotations"]:

            time = annotation["gameTime"]
            camera_change = annotation["change_type"]
            half = int(time[0])

            minutes = int(time[-5:-3])
            seconds = int(time[-2::])
            framerate = self.framerate
            frame = framerate * ( seconds + 60 * minutes )-1

            # Onehot for camera change
            if camera_change in self.dict_change:
                
                label_change = self.dict_change[camera_change]

                if half == 1:
                    frame = min(frame, feat_half1.shape[0]-1)
                    label_change_half1[frame] = 1

                if half == 2:
                    frame = min(frame, feat_half2.shape[0]-1)
                    label_change_half2[frame] = 1

        def feats2clip(feats, stride, clip_length, padding="replicate_last"):

            if padding == "zeropad":
                print("beforepadding", feats.shape)
                pad = feats.shape[0] - int(feats.shape[0]/stride)*stride
                print("pad need to be", clip_length-pad)
                m = torch.nn.ZeroPad2d((0, 0, clip_length-pad, 0))
                feats = m(feats)
                print("afterpadding", feats.shape)

            idx = torch.arange(start=0, end=feats.shape[0]-1, step=stride)
            idxs = []
            for i in torch.arange(0, clip_length):
                idxs.append(idx+i)
            idx = torch.stack(idxs, dim=1)

            if padding == "replicate_last":
                idx = idx.clamp(0, feats.shape[0]-1)

            return feats[idx,:]

        feat_half1 = feats2clip(torch.from_numpy(feat_half1), 
                        stride=self.chunk_size-self.receptive_field, 
                        clip_length=self.chunk_size)
        feat_half2 = feats2clip(torch.from_numpy(feat_half2), 
                        stride=self.chunk_size-self.receptive_field, 
                        clip_length=self.chunk_size)

        return feat_half1, feat_half2, torch.from_numpy(label_change_half1.copy()), torch.from_numpy(label_change_half2.copy()),torch.from_numpy(label_half1.copy()), torch.from_numpy(label_half2.copy())
        # return feat_half1, feat_half2, torch.from_numpy(label_change_half1), torch.from_numpy(label_change_half2)
    def __len__(self):
        return len(self.listGames)


class SoccerNet(Dataset):
    def __init__(self, path, features="ResNET_PCA512.npy", split="train", framerate=5):
        self.path = path
        self.listGames = getListGames(split, task="camera-changes")
        self.features = features
        self.framerate = framerate
        self.dict_change = Camera_Change_DICTIONARY
        self.labels = "Labels-cameras.json"

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            feat_half1 (np.array): features for the 1st half.
            feat_half2 (np.array): features for the 2nd half.
            label_half1 (np.array): labels (one-hot) for the 1st half.
            label_half2 (np.array): labels (one-hot) for the 2nd half.
        """
        # Load features
        feat_half1 = np.load(os.path.join(self.path, self.listGames[index], "1_" + self.features))
        feat_half2 = np.load(os.path.join(self.path, self.listGames[index], "2_" + self.features))

        # Load labels
        labels = json.load(open(os.path.join(self.path, self.listGames[index], self.labels)))

        label_change_half1 = np.zeros(feat_half1.shape[0])
        label_change_half2 = np.zeros(feat_half2.shape[0])

        for annotation in labels["annotations"]:

            time = annotation["gameTime"]
            camera_change = annotation["change_type"]
            half = int(time[0])

            minutes = int(time[-5:-3])
            seconds = int(time[-2::])
            framerate = self.framerate
            frame = framerate * ( seconds + 60 * minutes )-1

            # Onehot for camera change
            if camera_change in self.dict_change:
                
                label_change = self.dict_change[camera_change]

                if half == 1:
                    frame = min(frame, feat_half1.shape[0]-1)
                    label_change_half1[frame] = 1

                if half == 2:
                    frame = min(frame, feat_half2.shape[0]-1)
                    label_change_half2[frame] = 1

        # return feat_half1, feat_half2, label_change_half1, label_change_half2
        return torch.from_numpy(feat_half1), torch.from_numpy(feat_half2), torch.from_numpy(label_change_half1), torch.from_numpy(label_change_half2)

    def __len__(self):
        return len(self.listGames)
