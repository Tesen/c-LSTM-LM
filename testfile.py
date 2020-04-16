# -*- coding: utf-8 -*-
import sys
import json
import time
import datetime
import argparse
from data import SongLyricDataset#, collate_fn
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data as data
from torch.nn.utils.rnn import pack_padded_sequence
from collections import defaultdict
# from model import MCLLM


def main(args):
    """ Set the random seed manually for reproducibility """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    """ Load Data """
    data_set = SongLyricDataset(data=args.data, word_size=args.word_size, window=args.window)
    word_size = data_set.word_size
    feature_size = data_set.feature_size
    syllable_size = data_set.syllable_size
    lp.lprint("------ Data Stats -----", True)
    lp.lprint("{:>12}:  {}".format("song size", len(data_set)), True)
    lp.lprint("{:>12}:  {}".format("vocab size", word_size), True)
    lp.lprint("{:>12}:  {}".format("feature size", feature_size), True)
    lp.lprint("{:>12}:  {}".format("syllable size", syllable_size), True)
    print(data_set)

    """ write vocab and model param"""
    with open(args.checkpoint+".feature.json", 'w') as f:
        f.write(json.dumps(data_set.idx2feature, ensure_ascii=False))

    with open(args.checkpoint+".vocab.json", 'w') as f:
        f.write(json.dumps(data_set.idx2word, ensure_ascii=False))

    with open(args.checkpoint+".param.json", 'w') as f:
        f.write(json.dumps({"feature_idx_path":args.checkpoint+".feature.json", 
                            "vocab_idx_path":args.checkpoint+".vocab.json", 
                            "word_dim":args.word_dim, 
                            "melody_dim":args.melody_dim, 
                            "syllable_size":syllable_size, 
                            "feature_size":feature_size, 
                            "window":args.window, 
                            "args_word_size":args.word_size}, ensure_ascii=False))

    """
    # split train/valid 
    n_samples = len(data_set)
    train_size = int(len(data_set) * 0.9)
    val_size = n_samples - train_size
    train_data_set, val_data_set = torch.utils.data.random_split(data_set, [train_size, val_size])

    print(train_data_set)

    # Make data loader 
    train_data_loader = torch.utils.data.DataLoader(dataset=train_data_set, 
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers, 
                                              collate_fn=collate_fn)

    val_data_loader = torch.utils.data.DataLoader(dataset=val_data_set, 
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers, 
                                              collate_fn=collate_fn)

    # print("Type dataloader", type(train_data_loader))
    # print("Dataloader", train_data_loader)
    """
    