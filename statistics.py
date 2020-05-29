# -*- coding: utf-8 -*-
import os
import sys
import json
import argparse
import random
import math
import numpy as np

from midi_utils.convert_midi4generation import convert
from utils import repackage_hidden

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import defaultdict
from deeper_model import deepCLLM
from model import CLLM

from generate import generate, sample
from generate_deeper import generate_deeper, sample

def deeper(args, notes):
    with torch.no_grad():
        generated_lyrics, positions, score = generate_deeper(notes=notes, 
                                            param=args.deeper_param, checkpoint=args.deeper_checkpoint, 
                                            seed=args.seed, window=args.window, 
                                            temperature=args.temperature, LM_model = args.deeper_LM_model)
    for i, word in enumerate(generated_lyrics):
        print("word ", i, ": ", word)

def normal(args, notes):
    with torch.no_grad():
        generated_lyrics, positions, score = generate(notes=notes, 
                                            param=args.param, checkpoint=args.checkpoint, 
                                            seed=args.seed, window=args.window, 
                                            temperature=args.temperature, LM_model = args.LM_model)
    
    for i, word in enumerate(generated_lyrics):
        print("word ", i, ": ", word)

def main(args):
    argparse_dict = vars(args)
    print("------ Parameters -----")
    for k, v in argparse_dict.items():
        print("{:>16}:  {}".format(k, v))

    subfolders = os.listdir(args.data)
    list_of_song_nots = []
    song_notes = []
    print(subfolders[73:74])

    # Load data
    for subfolder in subfolders[70:71]:
        print("Currently loading data from: " + subfolder)
        subfolder_path = os.path.join(args.data, subfolder)
        files = os.listdir(subfolder_path)
        for file in files[0:5]:
            try:
                notes = np.load(os.path.join(subfolder_path, file), allow_pickle=True)
            except OSError as e:
                print("File %s could not be loaded. Skips file."%file)
                continue
            
            for note in notes:
                song_notes.append((note[2], note[3]))

            list_of_song_nots.append(song_notes)
            song_notes = []
    
    
    notes = convert(args.midi)
    # print(notes)
    
    notes = list_of_song_nots[0]
    print(notes)
    # normal(args, notes)
    deeper(args, notes)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """ Data parameter """
    parser.add_argument("-midi", "--midi", dest="midi", default="./c-LSTM-LM/sample_data/sample.midi", type=str, help="MIDI file")
    parser.add_argument("-output", "--output", dest="output", default="./c-LSTM-LM/test_output/", type=str, help="Output directory")
    parser.add_argument("-data", "--data", dest="data", default="./dataset_creation/augmented_dataset3_sorted", type=str, help="Data directory")

    """ Model parameter """
    parser.add_argument("-param", "--param", dest="param", default="./c-LSTM-LM/checkpoint_12052020_1500/model.param.json", type=str, help="Parameter file path")
    parser.add_argument("-checkpoint", "--checkpoint", dest="checkpoint", default="./c-LSTM-LM/checkpoint_12052020_1500/", type=str, help="Checkpoint file path")

    parser.add_argument("-deeper_param", "--deeper_param", dest="deeper_param", default="./c-LSTM-LM/checkpoint_15052020_1300/model.param.json", type=str, help="Parameter file path")
    parser.add_argument("-deeper_checkpoint", "--deeper_checkpoint", dest="deeper_checkpoint", default="./c-LSTM-LM/checkpoint_15052020_1300/", type=str, help="Checkpoint file path")

    """ Generation parameter """
    parser.add_argument("-seed", "--seed", dest="seed", default=0, type=int, help="Seed number for random library")
    parser.add_argument("-window", "--window", dest="window", default=20, type=int, help="Window size for beam search")
    parser.add_argument("-temperature", "--temperature", dest="temperature", default=1.0, type=float, help="Word sampling temperature")

    parser.add_argument("-LM_model", "--LM_model", dest="LM_model", default="model_25.pt", type=str, help="Model number of checkpoint")
    parser.add_argument("-deeper_LM_model", "--deeper_LM_model", dest="deeper_LM_model", default="model_15.pt", type=str, help="Model number of checkpoint")

    args = parser.parse_args()
    main(args)