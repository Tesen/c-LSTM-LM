# -*- coding: utf-8 -*-
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
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import defaultdict
from model import CLLM

import logging
logging.disable(logging.FATAL)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def generate(notes, param, checkpoint, seed=0, window=2, temperature=1.0):
    """ Set the random seed manually for reproducibility """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    """ Load parameters """
    params = json.loads(open(param, "r").readline())
    melody_dim = params["melody_dim"]
    word_dim = params["word_dim"]
    feature_window = params["window"]

    idx2feature = json.loads(open(params["feature_idx_path"], "r").readline())
    feature2idx = dict([(v, int(k)) for k, v in idx2feature.items()])
    feature_size = len(feature2idx)

    idx2word = json.loads(open(params["vocab_idx_path"], "r").readline())
    word2idx = dict([(v, int(k)) for k, v in idx2word.items()])
    word_size = len(word2idx)
    bb = word2idx["<BB>|<null>"]
    bl = word2idx["<BL>|<null>"]







    return generated, note_positions, score


def save_lyrics(generated, notes, output_dir):
    yomi = []
    yomi_w_word = []
    line = []


def main(args):
    argparse_dict = vars(args)
    print("------ Parameters -----")
    for k, v in argparse_dict.items():
        print("{:>16}:  {}".format(k, v))

    notes = convert(args.midi)
    with torch.no_grad():
        lyrics, positions, score = generate(notes=notes, 
                                            param=args.param, checkpoint=args.checkpoint, 
                                            seed=args.seed, window=args.window, 
                                            temperature=args.temperature)
    save_lyrics(lyrics, notes, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """ Data parameter """
    parser.add_argument("-midi", "--midi", dest="midi", default="./sample_data/sample.midi", type=str, help="MIDI file")
    parser.add_argument("-output", "--output", dest="output", default="./output/", type=str, help="Output directory")

    """ Model parameter """
    parser.add_argument("-param", "--param", dest="param", default="./checkpoint/model.param.json", type=str, help="Parameter file path")
    parser.add_argument("-checkpoint", "--checkpoint", dest="checkpoint", default="./checkpoint/model_05.pt", type=str, help="Checkpoint file path")

    """ Generation parameter """
    parser.add_argument("-seed", "--seed", dest="seed", default=0, type=int, help="Seed number for random library")
    parser.add_argument("-window", "--window", dest="window", default=20, type=int, help="Window size for beam search")
    parser.add_argument("-temperature", "--temperature", dest="temperature", default=1.0, type=float, help="Word sampling temperature")
    args = parser.parse_args()
    main(args)