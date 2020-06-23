# -*- coding: utf-8 -*-
import os
import sys
import json
import argparse
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import f1_score
from collections import Counter

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
from generate_deeper import save_lyrics, readable

import logging
logging.disable(logging.FATAL)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device type: %s"%device)

def get_correct_length(length):
    length = int(length)
    if length <= 0.25:
        return 0.25
    elif length <= 0.5:
        return 0.5
    elif length <= 0.75:
        return 0.75
    elif length <= 1:
        return 1.0
    elif length <= 1.5:
        return 1.5
    elif length <= 2:
        return 2.0
    elif length <= 3:
        return 3.0
    elif length <= 4:
        return 4.0
    elif length <= 6:
        return 6.0
    elif length <= 8:
        return 8.0
    elif length <= 16:
        return 16.0
    elif length <= 32:
        return 32.0

def deeper(args, list_of_song_notes, test_data, test_data_loader, batch_size, generate_lyrics=False, notes=None):
    """ Set the random seed manually for reproducibility """
    seed = 0
    temperature = args.temperature
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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

    """ Generate Lyrics and Statistics """
    if generate_lyrics:        
        # TODO: Create dictionary for note/rest boundary correlations
        note_types = ['note', 'rest']
        tags = ['<BB>', '<BL>', '<None>', '<Word>']
        lengths = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 3.5, 4.0, 6.0, 8.0, 16.0, 32.0]
        feature_boundary_cnt = []
        for note_type in note_types:
            for length in lengths:
                for tag in tags:
                    feature_boundary_cnt.append('%s[%s]%s'%(note_type, length, tag))

        og_feature_boundary_cnt = dict((f, 0) for f in feature_boundary_cnt)
        gen_feature_boundary_cnt = dict((f, 0) for f in feature_boundary_cnt)
        print(len(og_feature_boundary_cnt), og_feature_boundary_cnt)


        # Initiate syll counter arrays
        og_syll_per_line = []
        og_syll_per_block = []
        gen_syll_per_line = []
        gen_syll_per_block = []
        f1_bbs = []
        f1_bls = []
        f1_nones = []

        # For each song in list_of_song_notes. Generate predicted output, print both, create feature arrays and conduct analysis
        for notes in list_of_song_notes:
            song_notes = []

            # Create boundary feature vector to calculate F1-score
            original_boundary_features = []
            bl_syllcnt = 0
            bb_syllcnt = 0
            boundary_word = 0
            print('Original lyrics: ')
            for note in notes:
                print(note)
                word_idx = note[1]
                song_notes.append((note[2], note[3])) # Create note vector

                # Count note/rest boundary correlations dictionary
                if note[0] > 0 and note[0] < len(notes): # Skip first and last note
                    if note[2] == 'rest' and note[-2] != '<None>': # If rest witin word
                        og_feature_boundary_cnt['%s[%s]%s'%('rest', get_correct_length(note[3]), '<None>')] += 1
                    elif note[2] == 'rest' and note[-2] == '<None>': # If rest between words
                        og_feature_boundary_cnt['%s[%s]%s'%('rest', get_correct_length(note[3]), '<Word>')] += 1
                    elif note[-1] == '<None>': # If normal boundary
                        og_feature_boundary_cnt['%s[%s]%s'%('note', get_correct_length(note[3]), '<Word>')] += 1
                    else: # If BB or BL
                        og_feature_boundary_cnt['%s[%s]%s'%('note', get_correct_length(note[3]), note[-1])] += 1


                # Calculate number of syllables per line and block
                if note[2] != 'rest' and int(note[1]) > 0 and note[1] != boundary_word:
                    if note[-1] == '<BB>':
                        og_syll_per_block.append(bb_syllcnt)
                        bb_syllcnt = 0
                        
                        og_syll_per_line.append(bl_syllcnt)
                        bl_syllcnt = 0

                        boundary_word = note[1]
                    elif note[-1] == '<BL>':
                        og_syll_per_line.append(bl_syllcnt)
                        bl_syllcnt = 0
                        boundary_word = note[1]

                if note[2] != 'rest' and note[0]>0:
                    bb_syllcnt += 1
                    bl_syllcnt += 1

                if note[-1] == '<BB>':
                    original_boundary_features.append(word2idx['<BB>|<null>'])
                elif note[-1] == '<BL>':
                    original_boundary_features.append(word2idx['<BL>|<null>'])
                else:
                    original_boundary_features.append(0)
                
            og_syll_per_block.append(bb_syllcnt)
            og_syll_per_line.append(bl_syllcnt)
            print('og_syll_per_block: ', og_syll_per_block)
            print('og_syll_per_line: ', og_syll_per_line)


            # NOTE: Comment to not generate new lyrics and load old generated lyrics 
            # with torch.no_grad():
            #     generated_lyrics, positions, score = generate_deeper1(notes=song_notes, 
            #                                         param=args.deeper_param, checkpoint=args.deeper_checkpoint, 
            #                                         seed=args.seed, window=args.window, 
            #                                         temperature=args.temperature, LM_model = args.deeper_LM_model)
            # np.save('c-LSTM-LM/test_output/genearated_lyrics_stat', generated_lyrics)
            generated_lyrics = np.load('c-LSTM-LM/test_output/genearated_lyrics_stat.npy')

            pred_generated = readable(generated_lyrics, song_notes, args.checkpoint)
            
            # Create boundary feature vector to calculate F1-score
            predicted_boundary_fetures = []
            bb_syllcnt = 0
            bl_syllcnt = 0
            print('Generated lyrics: ')
            for note in pred_generated:
                # Count note/rest boundary correlations dictionary
                if note[0] > 0 and note[0] < len(notes): # Skip first and last note
                    if note[2] == 'rest' and note[-2] != '<None>': # If rest witin word
                        gen_feature_boundary_cnt['%s[%s]%s'%('rest', get_correct_length(note[3]), '<None>')] += 1
                    elif note[2] == 'rest' and note[-2] == '<None>': # If rest between words
                        gen_feature_boundary_cnt['%s[%s]%s'%('rest', get_correct_length(note[3]), '<Word>')] += 1
                    elif note[-1] == '<None>': # If normal boundary
                        gen_feature_boundary_cnt['%s[%s]%s'%('note', get_correct_length(note[3]), '<Word>')] += 1
                    else: # If BB or BL
                        gen_feature_boundary_cnt['%s[%s]%s'%('note', get_correct_length(note[3]), note[-1])] += 1

                # Calculate number of syllables per line and block
                if note[2] != 'rest' and int(note[1]) > 0 and note[1] != boundary_word: 
                    if note[-1] == '<BB>':
                        gen_syll_per_block.append(bb_syllcnt)
                        bb_syllcnt = 0
                        
                        gen_syll_per_line.append(bl_syllcnt)
                        bl_syllcnt = 0

                        boundary_word = note[1]
                    elif note[-1] == '<BL>':
                        gen_syll_per_line.append(bl_syllcnt)
                        bl_syllcnt = 0
                        boundary_word = note[1]

                if note[2] != 'rest' and note[0]>0:
                    bb_syllcnt += 1
                    bl_syllcnt += 1
                
                
                if note[-1] == '<BB>':
                    predicted_boundary_fetures.append(word2idx['<BB>|<null>'])
                elif note[-1] == '<BL>':
                    predicted_boundary_fetures.append(word2idx['<BL>|<null>'])
                else:
                    predicted_boundary_fetures.append(0)
            
            gen_syll_per_block.append(bb_syllcnt)
            gen_syll_per_line.append(bl_syllcnt)
            print('gen_syll_per_block: ', gen_syll_per_block)
            print('gen_syll_per_line: ', gen_syll_per_line)


            """ Calculate F1 score """
            f1 = f1_score(original_boundary_features, predicted_boundary_fetures, average=None)
            f1_bb = f1[1]
            f1_bl = f1[2]
            f1_none = f1[0]
            f1_bbs.append(f1_bb)
            f1_bls.append(f1_bl)
            f1_nones.append(f1_none)
    
    # Calculate, plot and save number of unique numbers in syll per line and block arrays
    def syll_dist_analysis():
        cnt_og_syll_per_block = Counter(og_syll_per_block)
        key_og_syll_per_block = sorted(cnt_og_syll_per_block.keys())
        val_og_syll_per_block = [cnt_og_syll_per_block[key] for key in key_og_syll_per_block]

        cnt_og_syll_per_line = Counter(og_syll_per_block)
        key_og_syll_per_line = sorted(cnt_og_syll_per_line.keys())
        val_og_syll_per_line = [cnt_og_syll_per_block[key] for key in key_og_syll_per_line]   

        cnt_gen_syll_per_block = Counter(gen_syll_per_block)
        key_gen_syll_per_block = sorted(cnt_gen_syll_per_block.keys())
        val_gen_syll_per_block = [cnt_gen_syll_per_block[key] for key in key_gen_syll_per_block]

        cnt_gen_syll_per_line = Counter(gen_syll_per_block)
        key_gen_syll_per_line = sorted(cnt_gen_syll_per_line.keys())
        val_gen_syll_per_line = [cnt_gen_syll_per_block[key] for key in key_gen_syll_per_line]

        plt.figure('Syll per block')
        plt.plot(key_og_syll_per_block, val_og_syll_per_block, 'k', label='Test songs')
        plt.plot(key_gen_syll_per_block, val_gen_syll_per_block, 'b', label='Generated songs')
        plt.show()
        plt.savefig(args.deeper_checkpoint + 'test/sylls_per_block.png')

        plt.figure('Syll per line')
        plt.plot(key_og_syll_per_line, val_og_syll_per_line, 'k', label='Test songs')
        plt.plot(key_gen_syll_per_line, val_gen_syll_per_line, 'b', label='Generated songs')
        plt.show()
        plt.savefig(args.deeper_checkpoint + 'test/sylls_per_line.png')
        
        with open(args.deeper_checkpoint + 'test/og_syll_per_block.json', 'w') as f:
            f.write(json.dumps(cnt_og_syll_per_block))
        
        with open(args.deeper_checkpoint + 'test/og_syll_per_line.json', 'w') as f:
            f.write(json.dumps(cnt_og_syll_per_line))

        with open(args.deeper_checkpoint + 'test/gen_syll_per_block.json', 'w') as f:
            f.write(json.dumps(cnt_gen_syll_per_block))

        with open(args.deeper_checkpoint + 'test/gen_syll_per_line.json', 'w') as f:
            f.write(json.dumps(cnt_gen_syll_per_line))

    # Save note/rest boundary correlations dictionaries  

    def boundary_dist_analysis(feature_boundary_cnt, dataset):
        with open(args.deeper_checkpoint + 'test/%s.json'%(dataset + 'feature_boundary_cnt'), 'w') as f:
            f.write(json.dumps(feature_boundary_cnt))

        print('feature_boundary_cnt: ',feature_boundary_cnt)
        N = len(feature_boundary_cnt)/8
        rest_len_boundary_matrix = np.zeros((int(N), 4))

        note_boundary_cnt = [0, 0, 0, 0]
        rest_boundary_cnt = [0, 0, 0, 0]

        items = list(feature_boundary_cnt.items())
        print('items: ', len(items), items)

        i = 0
        j = 0
        for item in items:
            if item[0].startswith('note'):
                if item[0].endswith('<BB>'):
                    note_boundary_cnt[0] += item[1]
                elif item[0].endswith('<BL>'):
                    note_boundary_cnt[1] += item[1]
                elif item[0].endswith('<None>'):
                    note_boundary_cnt[2] += item[1]
                elif item[0].endswith('<Word>'):
                    note_boundary_cnt[3] += item[1]

            elif item[0].startswith('rest'):
                if item[0].endswith('<BB>'):
                    rest_boundary_cnt[0] += item[1]
                elif item[0].endswith('<BL>'):
                    rest_boundary_cnt[1] += item[1]
                elif item[0].endswith('<None>'):
                    rest_boundary_cnt[2] += item[1]
                elif item[0].endswith('<Word>'):
                    rest_boundary_cnt[3] += item[1]
                

                if i > len(items)/2 and (i+1)%4 == 0:
                    bar = [items[i-3][1], items[i-2][1], items[i-1][1], items[i][1]]
                    if np.sum(bar) != 0:
                        rest_len_boundary_matrix[j] = bar/np.sum(bar)
                    else:
                        rest_len_boundary_matrix[j] = bar
                    j += 1
            i += 1

        rest_len_boundary_matrix = np.transpose(rest_len_boundary_matrix)
        xnames = lengths
        width = 0.35
        ind = np.arange(N)
        bbs = rest_len_boundary_matrix[0]
        bls = rest_len_boundary_matrix[1]
        nones = rest_len_boundary_matrix[2]
        words = rest_len_boundary_matrix[3]
        p1 = plt.bar(ind, bbs, width, color='b')
        p2 = plt.bar(ind, bls, width, color='r', bottom=bbs)
        p3 = plt.bar(ind, nones, width, color='g', bottom=list(map(lambda x,y: x+y, bbs,bls)))
        p4 = plt.bar(ind, words, width, color='y', bottom=list(map(lambda x,y,z: x+y+z, bbs,bls,nones)))
        plt.xticks(ind, xnames)
        plt.yticks(np.arange(0, 1.011, 0.1))

        blue_patch = mpatches.Patch(color='b', label='Block')
        red_patch = mpatches.Patch(color='r', label='Line')
        green_patch = mpatches.Patch(color='g', label='None')
        yellow_patch = mpatches.Patch(color='y', label='Word')
        plt.legend(handles=[blue_patch, red_patch, green_patch, yellow_patch])

        plt.show()

    syll_dist_analysis()
    boundary_dist_analysis(og_feature_boundary_cnt, 'og')
    boundary_dist_analysis(gen_feature_boundary_cnt, 'gen')
    

    


    """ Load model """
    model = deepCLLM(word_dim=word_dim, melody_dim=melody_dim, syllable_size=syllable_size, word_size=word_size, feature_size=feature_size, num_layers=3).to(device)
    path = args.deeper_checkpoint + args.deeper_LM_model
    cp = torch.load(path)
    model.load_state_dict(cp)

    sum_losses_lyric, sum_losses_syll =  evaluate(model, test_data, test_data_loader, batch_size)
    ppl_lyric = math.exp(sum_losses_lyric.avg)
    ppl_syll = math.exp(sum_losses_syll.avg)   

    lp.lprint('| Evaluation: '
                          '| Test Loss(Syllable) {loss_s.avg:5.5f} |'
                          '| Test Loss(Lyrics) {loss_l.avg:5.5f} |'
                          '| Test PPL(Syllable) {ppl_s:5.5f} |'
                          '| Test PPL(Lyrics) {ppl_l:5.5f} |' 
                          .format(loss_s=sum_losses_syll, 
                                  loss_l=sum_losses_lyric,
                                  ppl_s=ppl_syll,
                                  ppl_l=ppl_lyric), True)

    f1_bb = np.average(f1_bbs)
    f1_bl = np.average(f1_bls)
    f1_none = np.average(f1_none)
    lp.lprint('| Eval: '
                          '| F1-score (BB) {f1bb:5.5f} |'
                          '| F1-score (BL) {f1bl:5.5f} |'
                          '| F1-score (Word) {f1none:5.5f} |' 
                          .format(f1bb=f1_bb, 
                                  f1bl=f1_bl,
                                  f1none=f1_none), True)

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
    for subfolder in subfolders[0:1]: # NOTE: Data is limited
        print("Currently loading data from: " + subfolder)
        subfolder_path = os.path.join(args.data, subfolder)
        files = os.listdir(subfolder_path)
        for file in files[50:51]: # NOTE: Data is limited
            print("Song name: ", file.split('.')[0])
            try:
                notes = np.load(os.path.join(subfolder_path, file), allow_pickle=True)
            except OSError as e:
                print("File %s could not be loaded. Skips file."%file)
                continue

            print("Number of notes: ", len(notes))
            list_of_song_notes.append(notes) 

    word_size = 512
    window = 10
    limit_data = True
    """ Load test data """
    test_data_set = SongLyricDataset(args.data, word_size, window, limit_data)
    data_word_size = test_data_set.word_size
    data_feature_size = test_data_set.feature_size
    data_syllable_size = test_data_set.syllable_size
    
    # Print data stats
    lp.lprint("------ Test Data Stats -----", True)
    lp.lprint("{:>15}:  {}".format("Number of songs", len(test_data_set)), True)
    lp.lprint("{:>15}:  {}".format("vocab size", data_word_size), True)
    lp.lprint("{:>15}:  {}".format("feature size", data_feature_size), True)
    lp.lprint("{:>15}:  {}".format("syllable size", data_syllable_size), True)
    lp.lprint("-----------", True)
    batch_size = 32
    test_data_loader = torch.utils.data.DataLoader(dataset=test_data_set,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  collate_fn=collate_fn)

    # notes = convert(args.midi)
    # print(notes)
    
    # notes = list_of_song_notes[0]
    # print(notes)
    # normal(args, notes)

    generate_lyrics = True
    with torch.no_grad():
        deeper(args, list_of_song_notes, test_data_set, test_data_loader, batch_size, generate_lyrics)

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
parser.add_argument("-temperature", "--temperature", dest="temperature", default=2.0, type=float, help="Word sampling temperature")

parser.add_argument("-LM_model", "--LM_model", dest="LM_model", default="model_25.pt", type=str, help="Model number of checkpoint")
parser.add_argument("-deeper_LM_model", "--deeper_LM_model", dest="deeper_LM_model", default="model_06.pt", type=str, help="Model number of checkpoint") # Change model

args = parser.parse_args()
argparse_dict = vars(args)
print("------ Parameters -----")
for k, v in argparse_dict.items():
    print("{:>17}:  {}".format(k, v))
lp = LogPrint(args.deeper_checkpoint + 'test/test.log', True)



main()