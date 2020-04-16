# -*- coding: utf-8 -*-
import sys
import os
import random
import numpy as np
import torch
import torch.utils.data as data
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

class SongLyricDataset(data.Dataset):
    def __init__(self, data, word_size, window):
        """ Create feature vocab and index dictionaries """
        # Create feature vocabulary
        for i in range(window):
            for length in lengths:
                features.append('length[%s]=%s'%(i, length))
                features.append('length[%s]=%s'%(-(i+1), length))
        # NOTE: Do I need to add rest and note tags to the dataset? In that case, I must augment the dataset even further

        # Create index dictionaries for features
        sorted_features = sorted(features)
        self.feature2idx = dict((f, i) for i, f in enumerate(sorted_features))
        self.idx2feature = dict((i, f) for i, f in enumerate(sorted_features))
        self.feature_size = len(self.feature2idx)

        """ Load data and create word and syllable vocab """
        # Load data
        files = os.listdir(data)

        # Initialize word occurance dictionary
        word_dict = defaultdict(int)
        syll_dict = defaultdict(int)
        # For each song
        for file in files:
            song_features = np.load(os.path.join(data, file), allow_pickle=True)
            
            # For each word in song lyric increment word occurance dictionary
            for word in song_features[2]:                           
                word_lower = word[0].lower()
                word_dict[word_lower] += 1
                syll_dict[word_lower] += np.shape(word)[0]

        # Create index dictionaries for words
        self.word2idx = {}
        self.word2idx["<pad>"] = 0 # Padding token to fill batches
        self.word2idx["<unk>"] = 1 # Unknown token to replace rare words
        self.word2idx["<BB>|<null>"] = 2
        self.word2idx["<BL>|<null>"] = 3
        self.idx2word = {}
        self.idx2word[0] = "<pad>"
        self.idx2word[1] = "<unk>"
        self.idx2word[2] = "<BB>|<null>"
        self.idx2word[3] = "<BL>|<null>"
        
        idx = 4
        syllables = set()

        # Create word index embedding dictionaries
        for word, freq in sorted(word_dict.items(), key=lambda x:x[1], reverse=True)[:word_size:]: # Sort word_dict after frequency and limit size to word_size (size of our dictionary)
            # Add number of syllables for each word
            syllables.add(np.round(syll_dict[word]/freq))

            # Build word/index dictionaries
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            idx += 1

        self.word_size = len(self.word2idx)
        self.syllable_size = max(syllables) + 10

        print("word size: ", self.word_size)
        print("syllable size: ", self.syllable_size)

        self.feature_size = 1
        self.idx2lyrics = {}








    def __len__(self):
        return len(self.idx2lyrics)

    def __getitem__(self, idx):
        sample = 0
        return sample

