# -*- coding: utf-8 -*-
import os
import sys
import json
import argparse
import random
import math
import numpy as np

# import utils
from midi_utils.convert_midi4generation import convert
from utils import repackage_hidden
from utils import LogPrint
from utils import AverageMeter

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import defaultdict
from deeper_model import deepCLLM
from model import CLLM

from data2 import SongLyricDataset, collate_fn

# from generate import generate, sample
from generate_deeper import generate_deeper1, sample
from generate_deeper import save_lyrics

import logging
logging.disable(logging.FATAL)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device type: %s"%device)

def deeper(args, test_data, test_data_loader, batch_size, generate_lyrics=False, notes=None):
    """ Set the random seed manually for reproducibility """
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # if generate_lyrics:
    #     print("No dont generate")
    #     with torch.no_grad():
    #         generated_lyrics, positions, score = generate_deeper(notes=notes, 
    #                                             param=args.deeper_param, checkpoint=args.deeper_checkpoint, 
    #                                             seed=args.seed, window=args.window, 
    #                                             temperature=args.temperature, LM_model = args.deeper_LM_model)
    #     for i, word in enumerate(generated_lyrics):
    #         print("word ", i, ": ", word)

    #     save_lyrics(generated_lyrics, notes, args.output, args.checkpoint)
    
    """ Load parameters """
    params = json.loads(open(args.deeper_param, "r").readline())
    melody_dim = params["melody_dim"]
    word_dim = params["word_dim"]
    feature_window = params["window"]
    syllable_size = params["syllable_size"]

    # Load feature dict
    idx2feature = json.loads(open(args.deeper_checkpoint + "model.feature.json", "r").readline())
    feature2idx = dict([(v, int(k)) for k, v in idx2feature.items()]) # Reverse idx2feature
    feature_size = len(feature2idx)

    # Load word dict
    idx2word = json.loads(open(args.deeper_checkpoint + "model.vocab.json", "r").readline())
    word2idx = dict([(v, int(k)) for k, v in idx2word.items()]) # Reverse idx2word
    word_size = len(word2idx)

    """ Load model """
    model = deepCLLM(word_dim=word_dim, melody_dim=melody_dim, syllable_size=syllable_size, word_size=word_size, feature_size=feature_size, num_layers=3).to(device)
    path = args.deeper_checkpoint + args.deeper_LM_model
    cp = torch.load(path)
    model.load_state_dict(cp)

    sum_losses_lyric, sum_losses_syll =  evaluate(model, test_data, test_data_loader, batch_size)
    ppl_lyric = math.exp(sum_losses_lyric.avg)
    ppl_syll = math.exp(sum_losses_syll.avg)
    print("ppl_lyric = ", ppl_lyric)
    print("ppl_syll = ", ppl_syll)

    lp.lprint('| Evaluation: '
                          '| Test Loss(Syllable) {loss_s.avg:5.5f} |'
                          '| Test Loss(Lyrics) {loss_l.avg:5.5f} |'
                          '| Test PPL(Syllable) {ppl_s:5.5f} |'
                          '| Test PPL(Lyrics) {ppl_l:5.5f} |' 
                          .format(loss_s=sum_losses_syll, 
                                  loss_l=sum_losses_lyric,
                                  ppl_s=ppl_syll,
                                  ppl_l=ppl_lyric))

def normal(args, notes):
    with torch.no_grad():
        generated_lyrics, positions, score = generate(notes=notes, 
                                            param=args.param, checkpoint=args.checkpoint, 
                                            seed=args.seed, window=args.window, 
                                            temperature=args.temperature, LM_model = args.LM_model)
    
    for i, word in enumerate(generated_lyrics):
        print("word ", i, ": ", word)

def evaluate(model, test_data, test_data_loader, batch_size):
    model.eval()    
    sum_losses_syll = AverageMeter()
    sum_losses_lyric = AverageMeter()

    """ Build Optimizers """
    # lr = 0.001
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr) # lr = 0.001
    loss_criterion = nn.CrossEntropyLoss() # Combines LogSoftmax() and NLLLoss() (Negative log likelihood loss)

    hidden = model.init_hidden(batch_size)

    for i, (syllable, lyric, melody, lengths) in enumerate(test_data_loader):
        local_bs = lyric.size(0)
        if local_bs != batch_size:
            continue

        """ Move dataloaders to GPU """
        syllable = syllable.to(device)
        lyric = lyric.to(device)
        melody = melody.to(device).float()
        lengths = lengths.to(device)

        """ Remove first melody feature """
        melody = melody[:, 1:] # We dont really want to do this?

        """ Detach hidden layers """
        hidden = repackage_hidden(hidden) # Function from PyTorch NLP official example

        """ Feedforward """
        # Feedforward
        syllable_output, lyrics_output, hidden = model(lyric[:, :-1], melody, lengths, hidden)
        
        # Define packed padded targets
        target_syllable = pack_padded_sequence(syllable[:, 1:], lengths-1, batch_first=True)[0]
        target_lyrics = pack_padded_sequence(lyric[:, 1:], lengths-1, batch_first=True)[0]
        
        # Calculate and update Cross-Entropy loss
        loss_syllable = loss_criterion(syllable_output, target_syllable)
        sum_losses_syll.update(loss_syllable)

        loss_lyrics = loss_criterion(lyrics_output, target_lyrics)
        sum_losses_lyric.update(loss_lyrics)
    
    return sum_losses_lyric, sum_losses_syll


def main():    
    subfolders = os.listdir(args.data)
    list_of_song_notes = []
    song_notes = []

    # Load data
    # for subfolder in subfolders[0:1]:
    #     print("Currently loading data from: " + subfolder)
    #     subfolder_path = os.path.join(args.data, subfolder)
    #     files = os.listdir(subfolder_path)
    #     for file in files[4:5]:
    #         print("Song name: ", file.split('.')[0])
    #         try:
    #             notes = np.load(os.path.join(subfolder_path, file), allow_pickle=True)
    #         except OSError as e:
    #             print("File %s could not be loaded. Skips file."%file)
    #             continue
            
    #         for note in notes:
    #             song_notes.append((note[2], note[3]))

    #         list_of_song_notes.append(song_notes)
    #         song_notes = []

    word_size = 512
    window = 10
    limit_data = False
    """ Load test data """
    test_data_set = SongLyricDataset(args.data, word_size, window, limit_data)
    data_word_size = test_data_set.word_size
    data_feature_size = test_data_set.feature_size
    data_syllable_size = test_data_set.syllable_size
    
    # Print data stats
    lp.lprint("------ Test Data Stats -----", True)
    lp.lprint("{:>12}:  {}".format("Number of songs", len(test_data_set)), True)
    lp.lprint("{:>12}:  {}".format("vocab size", data_word_size), True)
    lp.lprint("{:>12}:  {}".format("feature size", data_feature_size), True)
    lp.lprint("{:>12}:  {}".format("syllable size", data_syllable_size), True)

    batch_size = 32
    test_data_loader = torch.utils.data.DataLoader(dataset=test_data_set,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  collate_fn=collate_fn)

    # notes = convert(args.midi)
    # print(notes)
    
    # notes = list_of_song_nots[0]
    # print(notes)
    # normal(args, notes)
    # test_data_set = None
    # test_data_loader = None

    with torch.no_grad():
        deeper(args, test_data_set, test_data_loader, batch_size,  generate_lyrics=False)

parser = argparse.ArgumentParser()
""" Data parameter """
parser.add_argument("-midi", "--midi", dest="midi", default="./c-LSTM-LM/sample_data/sample.midi", type=str, help="MIDI file")
parser.add_argument("-output", "--output", dest="output", default="./c-LSTM-LM/test_output/", type=str, help="Output directory")
parser.add_argument("-data", "--data", dest="data", default="./dataset_creation/augmented_dataset4_sorted_test", type=str, help="Data directory")

""" Model parameter """
parser.add_argument("-param", "--param", dest="param", default="./c-LSTM-LM/checkpoint_12052020_1500/model.param.json", type=str, help="Parameter file path")
parser.add_argument("-checkpoint", "--checkpoint", dest="checkpoint", default="./c-LSTM-LM/checkpoint_12052020_1500/", type=str, help="Checkpoint file path")

deeper_checkpoint = 'checkpoint_03062020_1030' + '/' # Change chekpoint folder
parser.add_argument("-deeper_param", "--deeper_param", dest="deeper_param", default="./c-LSTM-LM/" + deeper_checkpoint + "model.param.json", type=str, help="Parameter file path")
parser.add_argument("-deeper_checkpoint", "--deeper_checkpoint", dest="deeper_checkpoint", default="./c-LSTM-LM/" + deeper_checkpoint, type=str, help="Checkpoint file path")

""" Generation parameter """
parser.add_argument("-seed", "--seed", dest="seed", default=0, type=int, help="Seed number for random library")
parser.add_argument("-window", "--window", dest="window", default=20, type=int, help="Window size for beam search")
parser.add_argument("-temperature", "--temperature", dest="temperature", default=1.0, type=float, help="Word sampling temperature")

parser.add_argument("-LM_model", "--LM_model", dest="LM_model", default="model_25.pt", type=str, help="Model number of checkpoint")
parser.add_argument("-deeper_LM_model", "--deeper_LM_model", dest="deeper_LM_model", default="model_06.pt", type=str, help="Model number of checkpoint") # Change model

args = parser.parse_args()
argparse_dict = vars(args)
print("------ Parameters -----")
for k, v in argparse_dict.items():
    print("{:>16}:  {}".format(k, v))
lp = LogPrint(args.deeper_checkpoint + 'test/log.txt', True)



main()