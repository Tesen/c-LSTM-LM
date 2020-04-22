# -*- coding: utf-8 -*-
import os
import sys
import argparse
import json
import random
import numpy as np
from utils import load_settings
from data import SongLyricDataset, collate_fn
from model import CLMM
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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


def main():
    """ Set seeds """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    """ Load data """
    data_set = SongLyricDataset(data, word_size, window)
    data_word_size = data_set.word_size
    data_feature_size = data_set.feature_size
    data_syllable_size = data_set.syllable_size

    # Print data stats
    lp.lprint("------ Data Stats -----", True)
    lp.lprint("{:>12}:  {}".format("Number of songs", len(data_set)), True)
    lp.lprint("{:>12}:  {}".format("vocab size", data_word_size), True)
    lp.lprint("{:>12}:  {}".format("feature size", data_feature_size), True)
    lp.lprint("{:>12}:  {}".format("syllable size", data_syllable_size), True)

    """ Save vocab arrays and models to checkpoint """
    with open(checkpoint + '.feature.json', 'w') as f:
        f.write(json.dumps(data_set.idx2feature))

    with open(checkpoint + '.vocab.json', 'w') as f:
        f.write(json.dumps(data_set.idx2word))

    with open(checkpoint + '.param.json', 'w') as f:
        f.write(json.dumps({"feature_idx_path": checkpoint+'.feature.json',
                            "vocab_idx_path": checkpoint+'.vocab.json',
                            "word_dim": word_dim,
                            "syllable_size": data_syllable_size,
                            "melody_dim": melody_dim,
                            "feature_size": data_feature_size,
                            "window": window,
                            "args_word_size": word_size}))
    

    """ Split data into training and validation data """
    n_samples = len(data_set)
    train_size = int(n_samples*train_rate)
    validation_size = int((n_samples - train_size)/2)
    test_size = validation_size
    
    train_data_set, val_data_set, test_data_set = torch.utils.data.random_split(data_set, [train_size, validation_size, test_size])

    print("Training set: ", len(train_data_set), " songs, Validation set: ", len(val_data_set), " songs, "
          "Test set: ", len(test_data_set), " songs.")

    """ Create PyTorch dataloaders """
    train_data_loader = torch.utils.data.DataLoader(dataset=train_data_set,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=num_workers,
                                                    collate_fn=collate_fn)

    val_data_loader = torch.utils.data.DataLoader(dataset=val_data_set,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=num_workers, 
                                                  collate_fn=collate_fn)

    test_data_loader = torch.utils.data.DataLoader(dataset=test_data_set,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=num_workers, 
                                                  collate_fn=collate_fn)

    """ Load CLLM model """
    model = CLMM(word_dim=word_dim, melody_dim=melody_dim, syllable_size=data_syllable_size, word_size=data_word_size, feature_size=data_feature_size).to(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a conditional-LSTM language model to generate lyrics given melody")
    parser.add_argument('--settings_file', help='json file of settings, overrides everything else', type=str, default='settings.txt')
    parser.add_argument('-verbose', '--verbose', dest="verbose", default=1, type=int, help="verbose: 0 or 1")

    args = parser.parse_args()
    settings = vars(args)
    settings = load_settings(settings)
    
    print(settings["checkpoint"])

    if args.verbose == 1:
        lp = LogPrint(settings['checkpoint'] + '.log', True)
    else:
        lp = LogPrint(settings['checkpoint'] + '.log', False)

    # Print settings
    lp.lprint("------ Parameters -----", True)
    for (k, v) in settings.items():
        lp.lprint("{:>16}:  {}".format(k, v), True)
    
    # Log settings
    with open(settings['checkpoint']+'args.json', 'w') as f:
        f.write(json.dumps(settings))
    
    # Update local variables
    locals().update(settings)
    main()
