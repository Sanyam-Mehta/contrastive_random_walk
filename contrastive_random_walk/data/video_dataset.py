import os
import numpy as np
import json
import random
import math

import torch
import torch.utils.data as data
import random
import cv2
from contrastive_random_walk.utils import extract_patches_with_jitter, make_palindrome
from PIL import Image
import albumentations as A

import torchvision.transforms as transforms


class VideoList(data.Dataset):
    def __init__(self, 
                 filelist, 
                 clip_len, 
                 is_train=True, 
                 frame_gap=1, 
                 transform=None, 
                 random_clip=True,
                 ):
        
        """
        Args:
            filelist (str): Path to the filelist containing the video paths.
            clip_len (int): Number of frames in a clip.
            is_train (bool): Whether the dataset is for training or not.
            frame_gap (int): Number of frames between each clip.
            transform (callable, optional): A function/transform that takes in a TxHxWxC video
                and returns a transformed version.
            random_clip (bool): Whether to randomly sample clips from the video or not.
        """

        self.filelist = filelist
        self.clip_len = clip_len
        self.is_train = is_train
        self.frame_gap = frame_gap

        self.random_clip = random_clip
        self.transform = transform
        
        f = open(self.filelist, 'r')
        self.jpgfiles = []
        self.fnums = []

        for line in f:
            rows = line.split()
            jpgfile = rows[0]
            fnum = int(rows[1])

            self.jpgfiles.append(jpgfile)
            self.fnums.append(fnum)

        f.close()

    def __getitem__(self, index):
        index = index % len(self.jpgfiles)
        folder_path = self.jpgfiles[index]
        fnum = self.fnums[index]

        frame_gap = self.frame_gap
        startframe = 0
        
        readjust = False
        
        while fnum - self.clip_len * frame_gap < 0:
            frame_gap -= 1
            readjust = True

        if readjust:
            print('framegap adjusted to ', frame_gap, 'for', folder_path)
        
        diffnum = fnum - self.clip_len * frame_gap
        if self.random_clip:
            startframe = random.randint(0, diffnum)
        else:
            startframe = 0

        files = os.listdir(folder_path)
        files.sort(key=lambda x:int(x.split('.')[0]))
        
        img_patches = []
        
        # reading video
        for i in range(self.clip_len):
            idx = int(startframe + i * frame_gap)
            img_path = "%s/%s" % (folder_path, files[idx])

            # BGR -> RGB!!!
            img = cv2.imread(img_path)[:,:,::-1] #.astype(np.float32)  

            # Check how to tranform the image. This method might not be the best and could be slower.
            _, modified_patches = extract_patches_with_jitter(
                img,
                transforms=self.transform,
            )

            # appending nd array to a list
            img_patches.append(modified_patches)

        # transform the list into a palindrome:
        img_patches = make_palindrome(img_patches)

        img_patches = np.stack(img_patches)

        # img_patches has dimensions (2*clip_len, 49, 64, 64, 3) [2*T, NxN, H, W, C]

        # if self.transform is not None:
        #     imgs = self.transform(imgs)

        return img_patches

    def __len__(self):
        return len(self.jpgfiles)


class SingleVideoDataset(data.Dataset):
    def __init__(self, video, clip_len, fps_range=[1,1], n_clips=100000):
        self.video = video
        self.clip_len = clip_len
        self.fps = fps_range
        self.n_clips = n_clips
        
    def __getitem__(self, index):
        fps = np.random.randint(*self.fps)
        idx = np.random.randint(self.video.shape[0]//fps - self.clip_len)
        x = self.video[::fps][idx:idx+self.clip_len]
        return x
        
    def __len__(self):
        return self.n_clips