# ------------------------------------------------------------------------------
# Copyright (c) Sichuan University
# Licensed under the MIT License.
# Created by Yi LIN(yilin@scu.edu.cn), Tingting ZHANG
# ------------------------------------------------------------------------------



import os
import sys
import random

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np


class Acupoint(data.Dataset):

    def __init__(self, root_dir, file_list):
        '''
        :param root_dir: the root dir of this dataset, such as your_saved_dir/FAcupoint
        :param file_list: file lists in the protocols directory, such as train_100.txt
        '''
        self.root_dir = root_dir
        self.file_list = file_list

        #default image and label paths
        self.annotations = os.path.join(self.root_dir, 'protocols/annotations.txt')
        self.img_dir = os.path.join(self.root_dir, 'images')

        self.images = []
        self.labels = {}
        self.required_files = []

        self.preload()


    def preload(self):
        #load required files
        with open(os.path.join(self.root_dir, 'protocols', self.file_list), 'r', encoding='utf-8') as fr:
            for line in fr:
                self.required_files.append(line.strip())
        #read all images first
        images, labels = [],{}
        image_files = os.listdir(self.img_dir)
        for img_ in image_files:
            if not img_.endswith('.tif'):
                continue
            images.append(img_)

        #read labels
        with open(self.annotations, 'r', encoding='utf-8') as fr:
            #read the field name here
            fields = fr.readline()
            for line in fr:
                segs = line.strip().split(',')
                file_name = segs[0]
                labels[file_name] = segs

        common_files = set(images) & set(labels.keys())
        

        for file_ in common_files:
            #confirm the loaded file
            if file_ not in self.required_files:
                continue
            ffp = os.path.join(self.img_dir, file_)
            self.images.append(ffp)
            self.labels[ffp] = labels[file_]

        print('loaded {} images from {}'.format(len(self.images), self.root_dir))


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        full_labels = self.labels[image_path]

        scale = float(full_labels[1])

        center_w = float(full_labels[2])
        center_h = float(full_labels[3])
        center = torch.Tensor([center_w, center_h])

        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
        target = np.array([float(s) for s in full_labels[4:]]).reshape(-1,2)

        #further operations can be added here, such as image preprocessing, etc.

        meta = {'path':image_path, 'center':center, 'scale': scale}

        return img, target, meta





if __name__ == '__main__':
    rt, list = './FAcupoint', 'train_100.txt'
    if len(sys.argv) >=3:
        rt, list = sys.argv[1:]
    ds = Acupoint(rt, list)
    loader = data.DataLoader(ds, batch_size=32)
    for batch, samples in enumerate(loader):
        img, target, meta = samples
        #print(meta['path'], target)

