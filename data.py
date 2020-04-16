# -*- coding: utf-8 -*-
import sys
import random
import numpy as np
import torch
import torc.utils.data as data
import glob
import logging
from collections import defaultdict

logging.disable(logging.FATAL)

# Set sampling seed
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Initiate feature vector
features = []
tags = ['<WORD>', '<BB>', '<BL>']
lengths = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 16, 32]


songDataPath = '/dataset_creation/augmented_datset'

class SongLyricDataset(data.Dataset):
    def __init__(self, data, word_size, window):
        # Create feature vocabulary
        for i in range(window):
            for length in lengths:
                features.append('length[%s]=%s'%(i, length))
                features.append('length[%s]=%s'%(-(i+1), length))
        
        # NOTE: Do I need to add rest and note tags to the dataset? In that case, I must augment the dataset even further

        print("features: ", features) 




    def __len__(self):
        return len(self.idx2lyrics)

    def __getitem__(self, idx):
        sample = 0
        return sample

