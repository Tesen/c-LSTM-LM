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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import defaultdict
from model import CLLM

import logging
logging.disable(logging.FATAL)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def sample(preds, temperature=1.0):
    # Helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

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

    syllable_size = params["syllable_size"]

    # Load feature dict
    idx2feature = json.loads(open(checkpoint + "model.feature.json", "r").readline())
    feature2idx = dict([(v, int(k)) for k, v in idx2feature.items()])
    feature_size = len(feature2idx)

    # Load word dict
    idx2word = json.loads(open(checkpoint + "model.vocab.json", "r").readline())
    word2idx = dict([(v, int(k)) for k, v in idx2word.items()])
    word_size = len(word2idx)
    bb = word2idx["<BB>|<null>"]
    bl = word2idx["<BL>|<null>"]

    """ Load model """
    model = CLLM(word_dim=word_dim, melody_dim=melody_dim, syllable_size=syllable_size, word_size=word_size, feature_size=feature_size, num_layers=1).to(device)
    model.load_state_dict(torch.load(checkpoint + "model_15.pt"))
    model.eval()
    hidden = model.init_hidden(1)

    """ define feature function """
    def create_feature(prev_notes, next_notes, prev_tag):
        feature_str = []
        feature_str.append("prev_tag=%s"%prev_tag)
        for j, pn in enumerate(prev_notes):
            if pn[0] == "rest":
                note_num = "note[-%s]=rest"%(len(prev_notes)-j)
            else:
                note_num = "note[-%s]=note"%(len(prev_notes)-j)
            note_duration = "length[-%s]=%s"%(len(prev_notes)-j, pn[1])
            feature_str.append(note_num)
            feature_str.append(note_duration)
        for j, nn in enumerate(next_notes):
            if nn[0] == "rest":
                note_num = "note[%s]=rest"%j
            else:
                note_num = "note[%s]=note"%j
            note_duration = "length[%s]=%s"%(j, nn[1])
            feature_str.append(note_num)
            feature_str.append(note_duration)
        feature = [feature2idx[f] for f in feature_str]
        return feature

    def update_note_position(notes, note_position, word):
        """
        - Move notes for the syllable count of the generated word.
        - Note that rest does not move.
            - if num[0]=rest, position+=1
        """
        if word.startswith(("<BOL>", "<BOB>")):
            pass
        else:
            word_length = int(len([y for y in word.split("|")[-1].split("_") if y != 'ッ']))       
            if word_length != 0:
                step = 0
                width = 0
                for n in notes[note_position::]:
                    num = n[0]
                    width += 1
                    if num == "rest":
                        pass
                    else:
                        step += 1
                    if step == word_length:
                        break
                note_position += width
        if note_position+1 >= len(notes):
            return note_position
        if notes[note_position][0] == 'rest' :
            note_position += 1
        return note_position

    """ Generate lyrics """
    accepted_lyrics = []
    max_nr_words = len(notes) - 1
    prob_forward = [[] for l in range(max_nr_words)]

    for t in range(max_nr_words):
        if t == 0: # If first word
            old_path = (word2idx["<BB>|<null>"], )
            old_note_positions = (0, )
            old_generated = dict(zip(old_note_positions, old_path))

            lengths = torch.Tensor([1]).long().to(device)

            # 1. Create word input vector
            x_word = torch.Tensor([[old_path[0]]]).long().to(device)

            # 2. Create melody input vector
            x_midi = np.zeros((1, t+1, feature_size)) # Initiate feature matrix
            i = t + 1

            # Create window of notes
            # This defines and lists the window for previous and upcoming notes
            prev_i = i - feature_window + 1
            if prev_i < 0: 
                prev_i = 0
            prev_notes = notes[prev_i:i]

            next_i = i + feature_window
            if next_i > len(notes):
                next_i = len(notes)
            next_notes = notes[i:next_i]

            # Create feature vector of previous 10 and next 10 notes
            feature_vec = create_feature(prev_notes, next_notes, "<BB>")

            for f in feature_vec:
                x_midi[0, 0, f] = 1 # No melody input, only MIDI value 1
            
            x_midi = torch.Tensor(x_midi).to(device)

            # 3. Generate word
            hidden = repackage_hidden(hidden)
            syllable_output, lyrics_output, hidden = model(x_word, x_midi, lengths + 1)

            # Apply softmax layer to text output
            dist = nn.functional.softmax(lyrics_output, dim=1).cpu().numpy()[0]
            dist[word2idx["<unk>"]] = 0.0
            # print("First dist: ", type(dist))
            stack = set()

            for x in range(100*window):
                new_index = sample(dist, temperature) # Sample word from probabiltity distribution of words
                new_word = idx2word[str(new_index)]
                new_path = old_path + (new_index, )
                new_note_positions = old_note_positions + (update_note_position(notes, old_note_positions[-1], new_word), )
                prob = math.log(dist[new_index])
                stack.add((new_path, new_note_positions, prob))
                if len(stack) >= window:
                    break
            
            for new in stack:
                prob_forward[t].append(new)
        else:
            # Initiate matrices
            x_word = np.zeros((window, t+1), dtype="int32")
            x_midi = np.zeros((window, t+1, feature_size))

            for y, (old_path, old_note_positions, old_prob) in enumerate(prob_forward[t-1]):
                # 1. Create word input vector
                for old_t, old_index in enumerate(old_path):
                    x_word[y, old_t] = old_index

                # 2. Create melody input vector
                if old_index == bb:
                    prev_tag = "<BB>"
                elif old_index == bl:
                    prev_tag = "<BL>"
                else:
                    prev_tag = "<WORD>"

                for old_t, old_midi_position in enumerate(old_note_positions):
                    i = old_midi_position

                    # Create window of notes
                    # This defines and lists the window for previous and upcoming notes
                    prev_i = i - feature_window + 1
                    if prev_i < 0: 
                        prev_i = 0
                    prev_notes = notes[prev_i:i]

                    next_i = i + feature_window
                    if next_i > len(notes):
                        next_i = len(notes)
                    next_notes = notes[i:next_i]

                    # Create feature vector of previous 10 and next 10 notes
                    feature_vec = create_feature(prev_notes, next_notes, prev_tag)

                    for f in feature_vec:
                        x_midi[y, old_t, f] = 1 # No melody input, only MIDI value 1

            x_word = torch.Tensor(x_word).long().to(device)
            x_midi = torch.Tensor(x_midi).to(device)
            lengths = torch.Tensor([t+1]*window).long().to(device)

            # 3. Generate word
            hidden = repackage_hidden(hidden)
            syllable_output, lyrics_output, hidden = model(x_word, x_midi, lengths + 1)

            # We only want the last output
            lyrics_output = lyrics_output[-window::]
            dists = nn.functional.softmax(lyrics_output, dim=1).cpu().numpy()
            # print("Second dists: ", type(dists), np.shape(dists))
            stack = set()
            for y in range(len(prob_forward[t-1])):
                dist = dists[y]
                # print("Third dist: ", type(dist), np.shape(dist))
                dist[word2idx["<unk>"]] = 0.0
                old_path = prob_forward[t-1][y][0]

                if old_path[-1] in (bb, bl):
                    new_segment = True
                else:
                    new_segment = False

                old_note_positions = prob_forward[t-1][y][1]
                old_prob = prob_forward[t-1][y][2]
                old_generated = dict(zip(old_note_positions, old_path))

                generate_bb = False
                generate_bl = False

                temp_stack = set()
                for z in range(100*window):
                    new_index = sample(dist, temperature)
                    if new_segment and new_index in (bb, bl):
                        continue
                    if new_index == bb:
                        generate_bb = True
                    if new_index == bl:
                        generate_bl = True

                    new_word = idx2word[str(new_index)]
                    new_path = old_path + (new_index, )
                    new_note_positions = old_note_positions + (update_note_position(notes, old_note_positions[-1], new_word), )
                    prob = math.log(dist[new_index]) +  old_prob
                    temp_stack.add((new_path, new_note_positions, prob))
                    if len(temp_stack) >= window:
                        break

                if not new_segment:
                    if generate_bb and not generate_bl:
                        new_word = "<BB>|<null>"
                        new_word = idx2word[str(new_index)]
                        new_path = old_path + (new_index, )
                        new_note_positions = old_note_positions + (update_note_position(notes, old_note_positions[-1], new_word), )
                        prob = math.log(dist[new_index]) +  old_prob
                        temp_stack.add((new_path, new_note_positions, prob))

                    elif not generate_bb and generate_bl:
                        new_word = "<BL>|<null>"
                        new_word = idx2word[str(new_index)]
                        new_path = old_path + (new_index, )
                        new_note_positions = old_note_positions + (update_note_position(notes, old_note_positions[-1], new_word), )
                        prob = math.log(dist[new_index]) +  old_prob
                        temp_stack.add((new_path, new_note_positions, prob))

                for item in temp_stack:
                    stack.add(item)

            count = 0
            for path, note_positions, prob in sorted(list(stack), key=lambda x:x[2], reverse=True):
                if note_positions[-1] >= max_nr_words:
                    entropy = -prob/(len(path) - 1)
                    accepted_lyrics.append((path, note_positions, entropy))
                elif note_positions[-1] < max_nr_words:
                    prob_forward[t].append((path, note_positions, prob))
                    count += 1
                if count >= window:
                    break

            if len(stack) > 0:
                last_max_pos = max([W[1][-1] for W in list(stack)])
                progress_num = 100*last_max_pos/len(notes)
                progress_bar = "Generate lyrics [" + "=" * int(progress_num/5) + ">" + "-" * (20 - int(progress_num/5)) + "]"
                sys.stderr.write("\r%s (%.1f %%)"%(progress_bar, progress_num))
                    
                
        # 4. Break
        if len(accepted_lyrics) >= 10:
            sys.stderr.write("\n")
            break
                        
    # 5. Determine and return output        
    path, note_positions, score = max(accepted_lyrics, key=lambda x:x[2])
    generated = [idx2word[str(idx)] for idx in path[1::]]
                   
    return generated, note_positions, score


def save_lyrics(generated, notes, output_dir):
    
    out_file = open(output_dir + 'output.txt', 'w')

    bb = "<BB>|<null>"
    bl = "<BL>|<null>"
    
    line = []
    for word in generated:
        if word == bl:
            out_file.write(" ".join(line) + '\n') # Write line boundary
            line = []
        if word == bb:
            out_file.write(" ".join(line) + '\n\n') # Write block boundary
        else:
            line.append(word)
        
    if len(line) > 0:
        out_file.write(" ".join(line) + '\n')
            
    


def main(args):
    argparse_dict = vars(args)
    print("------ Parameters -----")
    for k, v in argparse_dict.items():
        print("{:>16}:  {}".format(k, v))

    notes = convert(args.midi)
    with torch.no_grad():
        generated_lyrics, positions, score = generate(notes=notes, 
                                            param=args.param, checkpoint=args.checkpoint, 
                                            seed=args.seed, window=args.window, 
                                            temperature=args.temperature)
    save_lyrics(generated_lyrics, notes, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """ Data parameter """
    parser.add_argument("-midi", "--midi", dest="midi", default="./c-LSTM-LM/sample_data/sample.midi", type=str, help="MIDI file")
    parser.add_argument("-output", "--output", dest="output", default="./c-LSTM-LM/output/", type=str, help="Output directory")

    """ Model parameter """
    parser.add_argument("-param", "--param", dest="param", default="./c-LSTM-LM/checkpoint_30042020_1800/model.param.json", type=str, help="Parameter file path")
    parser.add_argument("-checkpoint", "--checkpoint", dest="checkpoint", default="./c-LSTM-LM/checkpoint_30042020_1800/", type=str, help="Checkpoint file path")

    """ Generation parameter """
    parser.add_argument("-seed", "--seed", dest="seed", default=0, type=int, help="Seed number for random library")
    parser.add_argument("-window", "--window", dest="window", default=20, type=int, help="Window size for beam search")
    parser.add_argument("-temperature", "--temperature", dest="temperature", default=1.0, type=float, help="Word sampling temperature")
    args = parser.parse_args()
    main(args)