# -*- coding: utf-8 -*-
import sys
import os
import json
import time
from datetime import datetime
import argparse
from data2 import SongLyricDataset, collate_fn
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
    
class LogPrint:
    def __init__(self, file_path, err):
        self.file = open(file_path, "w", buffering=1)
        self.err = err

    def lprint(self, text, ret=False, ret2=False):
        if self.err:
            if ret == True:
                if ret2 == True:
                    sys.stderr.write("\n" + text + "\n")
                else:
                    sys.stderr.write("\r" + text + "\n")
            else:
                sys.stderr.write("\r" + text)
        self.file.write(text + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """ Data parameter """
    parser.add_argument("-data", "--data", dest="data", default="./c-LSTM-LM/test_dataset", type=str, help="alignment data")
    date = datetime.date(datetime.now())
    checkpoint = './c-LSTM-LM/test-checkpoint' + str(date)+ 'v1'
    bol = True
    while bol:
        try:
            if not os.path.exists(checkpoint):
                os.mkdir(checkpoint)
            bol = False
        except FileExistsError:
            ver = int(checkpoint[-1]) + 1
            checkpoint = list(checkpoint)
            checkpoint[-1] = str(ver)
            checkpoint = ''.join(checkpoint)

    parser.add_argument("-checkpoint", "--checkpoint", dest="checkpoint", default=checkpoint, type=str, help="save path")

    """ Feature parameter """
    parser.add_argument("-window", "--window", dest="window", default=10, type=int, help="window size of melody featrure (default: 10)")
    parser.add_argument("-word_size", "--word_size", dest="word_size", default=20000, type=int, help="vocab size (default: 20000)")

    """ Model parameter """
    parser.add_argument("-word_dim", "--word_dim", dest="word_dim", default=512, type=int, help="dimension of Word Embedding (default: 512)")
    parser.add_argument("-melody_dim", "--melody_dim", dest="melody_dim", default=256, type=int, help="dimension of Melody Layer (default: 256)")

    """ Training parameter """
    parser.add_argument("-seed", "--seed", dest="seed", default=0, type=int, help="random seed")
    parser.add_argument("-num_workers", "--num_workers", dest="num_workers", default=4, type=int, help="number of CPU")
    parser.add_argument("-num_epochs", "--num_epochs", dest="num_epochs", default=5, type=int, help="Epochs")
    parser.add_argument("-batch_size", "--batch_size", dest="batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("-lr", "--lr", dest="lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("-log_interval", "--log_interval", dest="log_interval", default=10, type=int, help="Report interval")

    """ Logging parameter """
    parser.add_argument("-verbose", "--verbose", dest="verbose", default=1, type=int, help="verbose 0/1")
    args = parser.parse_args()

    
    """ Save parameter """
    if args.verbose == 1:
        lp = LogPrint(checkpoint + "/model" + ".log", True)
    else:
        lp = LogPrint(checkpoint + "/model" + ".log", False)
    argparse_dict = vars(args)
    lp.lprint("------ Parameters -----", True)
    for k, v in argparse_dict.items():
        lp.lprint("{:>16}:  {}".format(k, v), True)
    with open(args.checkpoint+".args.json", 'w') as f:
        f.write(json.dumps(argparse_dict))
    main(args)
