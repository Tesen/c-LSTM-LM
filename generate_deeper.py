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
from deeper_model import deepCLLM

import logging
logging.disable(logging.FATAL)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

def sample(preds, temperature=1.0):
    # Helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_deeper1(notes, param, checkpoint, seed=0, window=2, temperature=1.0, LM_model = ""):
    """ Set the random seed manually for reproducibility """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    """ Load parameters """
    params = json.loads(open(param, "r").readline())
    melody_dim = params["melody_dim"]
    word_dim = params["word_dim"]
    feature_window = params["window"]

    syllable_size = params["syllable_size"]

    # Load feature dict
    idx2feature = json.loads(open(checkpoint + "model.feature.json", "r").readline())
    print("idx2feature: ", idx2feature)
    feature2idx = dict([(v, int(k)) for k, v in idx2feature.items()]) # Reverse idx2feature
    feature_size = len(feature2idx)

    # Load word dict
    idx2word = json.loads(open(checkpoint + "model.vocab.json", "r").readline())
    word2idx = dict([(v, int(k)) for k, v in idx2word.items()]) # Reverse idx2word
    word_size = len(word2idx)
    bb = word2idx["<BB>|<null>"]
    bl = word2idx["<BL>|<null>"]

    # Load syllable counts
    word2syllablecnt = json.loads(open(checkpoint + "model.syllables.json", 'r').readline())
    print("word2syllablecnt: ", word2syllablecnt)

    """ Load model """
    model = deepCLLM(word_dim=word_dim, melody_dim=melody_dim, syllable_size=syllable_size, word_size=word_size, feature_size=feature_size, num_layers=3).to(device)
    print("LOAD PATH: ", checkpoint + LM_model)
    cp = torch.load(checkpoint + LM_model)
    model.load_state_dict(cp)
    model.eval()
    print('Model succesfully loaded!')
    hidden = model.init_hidden(1)

    """ Define feature function """
    def create_feature(prev_notes, next_notes, prev_tag):
        feature_str = []
        feature_str.append("prev_tag=%s"%prev_tag)
        for j, pn in enumerate(prev_notes):
            if pn[0] == "rest":
                note_num = "note[-%s]=rest"%(len(prev_notes)-j)
            else:
                note_num = "note[-%s]=note"%(len(prev_notes)-j)
            note_duration = "length[-%s]=%s"%(len(prev_notes)-j, get_correct_length(float(pn[1])))
            feature_str.append(note_num)
            feature_str.append(note_duration)
        for j, nn in enumerate(next_notes):
            if nn[0] == "rest":
                note_num = "note[%s]=rest"%j
            else:
                note_num = "note[%s]=note"%j
            note_duration = "length[%s]=%s"%(j, get_correct_length(float(nn[1])))
            feature_str.append(note_num)
            feature_str.append(note_duration)
        feature = [feature2idx[f] for f in feature_str]
        return feature

    """ Define function to update note position """
    def update_note_position(notes, note_position, word):
        """
        - Move notes for the syllable count of the generated word.
        - Note that rest does not move.
            - if num[0]=rest, position+=1
        """
        if word.startswith(("<BL>", "<BB>")):
            pass
        else:
            # try:
            #     word_length = int(word2syllablecnt[word]) 
            # except KeyError:
            #     print("\nKeyError for word: ", word)
            #     word_length = 0

            word_length = int(len([s for s in word.split('|')[-1].split('_')]))
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

    for t in range(max_nr_words): # For each note
        if t == 0: # If first note
            old_path = (word2idx["<BB>|<null>"], )
            old_note_positions = (0, )
            old_generated = dict(zip(old_note_positions, old_path))

            lengths = torch.Tensor([1]).long().to(device)

            # 1. Create word input tensor
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
                x_midi[0, 0, f] = 1
            x_midi = torch.Tensor(x_midi).to(device)

            # 3. Generate word
            syllable_output, lyrics_output, hidden = model(x_word, x_midi, lengths + 1, hidden)

            # Apply softmax layer to text output
            dist = nn.functional.softmax(lyrics_output, dim=1).cpu().numpy()[0]
            dist[word2idx["<unk>"]] = 0.0
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
        else: # Not first note
            # Initiate matrices
            x_word = np.zeros((window, t+1), dtype="int32")
            x_midi = np.zeros((window, t+1, feature_size))

            # For each probability sequence
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

                # For each previous note
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
                        x_midi[y, old_t, f] = 1 
    
            x_word = torch.Tensor(x_word).long().to(device)
            x_midi = torch.Tensor(x_midi).to(device)
            lengths = torch.Tensor([t+1]*window).long().to(device)

            # 3. Generate word
            hidden = model.init_hidden(20)
            syllable_output, lyrics_output, hidden = model(x_word, x_midi, lengths + 1, hidden) # Forward pass

            # We only want the last output
            lyrics_output = lyrics_output[-window::]

            # Apply softmax to output
            dists = nn.functional.softmax(lyrics_output, dim=1).cpu().numpy()
            stack = set()
            for y in range(len(prob_forward[t-1])):
                dist = dists[y] # Generated probability distributions
                dist[word2idx["<unk>"]] = 0.0

                # Create path for this generated distribution
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
                    new_index = sample(dist, temperature) # Sample from distribution
                    if new_segment and new_index in (bb, bl):
                        continue
                    if new_index == bb:
                        generate_bb = True
                    if new_index == bl:
                        generate_bl = True

                    new_word = idx2word[str(new_index)]
                    new_path = old_path + (new_index, )
                    new_note_positions = old_note_positions + (update_note_position(notes, old_note_positions[-1], new_word), )
                    
                    # Calculate probability of this specific path
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

            # Find most probable path of all generated distributions
            count = 0
            for path, note_positions, prob in sorted(list(stack), key=lambda x:x[2], reverse=True):
                if note_positions[-1] >= max_nr_words: # If last note
                    entropy = -prob/(len(path) - 1)
                    accepted_lyrics.append((path, note_positions, entropy)) # Save complete lyrics
                elif note_positions[-1] < max_nr_words: # If not last note
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
        if len(accepted_lyrics) >= 2: # Limit number of probability sequences in beam search, prune the rest
            sys.stderr.write("\n")
            break
                        
    # 5. Determine and return output        
    path, note_positions, score = max(accepted_lyrics, key=lambda x:x[2]) # Pick most probable path in beam search
    generated = [idx2word[str(idx)] for idx in path[1::]]
                   
    return generated, note_positions, score

def save_lyrics_old(generated, notes, output_dir, checkpoint):
    
    out_file = open(output_dir + 'output.txt', 'w')
    lyrics_file = open(output_dir + 'lyrics.txt', 'w')

    # Load syllable countss
    # word2syllablecnt = json.loads(open(checkpoint + "model.syllables.json", 'r').readline())
    # print(word2syllablecnt)

    words = []
    word_vec = []
    line = []
    bbs = set()
    bbs.add(0)
    bls = set()
    
    word_idx = 0
    temp_syllcnt = 1
    # Save txt file
    for word in generated:
        print("Word: ", word)
        if word.startswith("<BL>"):
            out_file.write(" ".join(line) + '\n') # Write line boundary
            lyrics_file.write(" ".join(w.split('|')[0] for w in line) + '\n')
            bls.add(word_idx)
            line = []
        elif word.startswith("<BB>"):
            out_file.write(" ".join(line) + '\n\n') # Write block boundary
            lyrics_file.write(" ".join(w.split('|')[0] for w in line) + '\n\n')
            line = []
            bbs.add(word_idx)
        else:
            # syllcnt = word2syllablecnt[word]
            # print("Syllcnt = ", (word, syllcnt))
            
            # if temp_syllcnt < syllcnt:
            #     temp_syllcnt += 1
            # elif temp_syllcnt == syllcnt:
            #     line.append(word)
            #     word_idx += 1
            #     temp_syllcnt = 0
            line.append(word)

            word_idx += 1
            words.append(word)
            word_vec.append([word, word_idx])
    
    # for (word, word_idx) in word_vec:
    #     print("Word_vec: ", (word_idx, word))

    if len(line) > 0:
        out_file.write(" ".join(line) + '\n')
        lyrics_file.write(" ".join(w.split('|')[0] for w in line) + '\n')

    # Save note feature array file
    song = {'lyrics': []}
    temp_lyrics = []

    print("\nlen(note): ", len(notes))
    print("len(word_vec): ", len(word_vec))

    cnt = 0
    
    for note_idx, (num, length) in enumerate(notes):
        length = int(float(length))
        
        if cnt >= len(word_vec):
            break

        word = word_vec[cnt][0]
        word_idx = word_vec[cnt][1]

        # note = [note_number, word_index, note_type, duration, word, syllable, feature_type]
        # note[0] = note_number, note[1] = word_index, note[2] = note_type(rest) or MIDI_number, note[3] = duration 
        # note[4] = word, note[5] = syllable, note[6] = [all syllables], note[7]= feature_type

        if num == 'rest':
            temp_lyrics.append([note_idx, '<None>', num, length, '<None>', '<None>', '<None>', '<None>'])
        else:
            if note_idx in bbs:
                temp_lyrics.append([note_idx, word_idx, num, length, word, [], '<BB>'])
            elif note_idx in bls:
                temp_lyrics.append([note_idx, word_idx, num, length, word, [], '<BL>'])
            else:
                temp_lyrics.append([note_idx, word_idx, num, length, word, [], '<WORD>'])

            cnt += 1

        print("temp_lyrics: ",  temp_lyrics[-1])
    

    last_note_idx = len(temp_lyrics) - 1
    for i, note in enumerate(temp_lyrics):
        if note[1] == 'rest':
            if i == 0 or i == last_note_idx: 
                song['lyrics'].append([note[0], '<None>', note[2], '<None>', '<None>', '<None>'])
            else:
                if temp_lyrics[i-1][-2] == temp_lyrics[i+1][-2]: # If new word
                    print("test: ", note[::] + ['<None>'] + temp_lyrics[i+1][-3::])
                    song['lyrics'].append(note[::] + ['<None>'] + temp_lyrics[i+1][-3::])
                else:
                    song['lyrics'].append([note[0], note[1], note[2], "<None>", "<None>", "<None>", "<None>"])
        else:
            song["lyrics"].append(note[::])
    song["lyrics"].append([last_note_idx+1, "rest", "32", "<None>", "<None>", "<None>", "<None>"])
        
    with open(output_dir + 'output.readable', 'w') as f:
        for note in song["lyrics"]:
            print(note)
            f.write("%s\n" %note)

def save_lyrics(generated, notes, output_dir, checkpoint):
    # Save word and lyrics file
    out_file = open(output_dir + 'output.txt', 'w')
    lyrics_file = open(output_dir + 'lyrics.txt', 'w')

    line = []
    syllables = []
    syll_with_word = []
    word_idx = 0
    bb_idxs = set()
    bb_idxs.add(0)
    bl_idxs = set()

    for word in generated:
        print(word)
        if word == '<BB>|<null>':
            lyrics_file.write(" ".join(w.split('|')[0] for w in line) + '\n\n')
            out_file.write(" ".join(line) + '\n\n')
            line = []
            bb_idxs.add(word_idx)
        elif word == '<BL>|<null>':
            lyrics_file.write(" ".join(w.split('|')[0] for w in line) + '\n')
            out_file.write(" ".join(line) + '\n')
            line = []
            bl_idxs.add(word_idx)
        else:
            line.append(word)
            syllables += word.split('|')[-1].split('_')
            syll_with_word += [(s, word_idx, word) for s in word.split('|')[-1].split('_')]
            word_idx += 1

    if len(line) > 0:
        lyrics_file.write(" ".join(w.split('|')[0] for w in line) + '\n')
        out_file.write(" ".join(line) + '\n')

    # Save note format file
    song = {'lyrics':[]}
    word_pos = 0
    temp_lyrics = []

    # note = [note_number, word_index, note_type, duration, word, syllable, feature_type]
    # note[0] = note_number, note[1] = word_index, note[2] = note_type(rest) or MIDI_number, note[3] = duration 
    # note[4] = word, note[5] = syllable, note[6] = [all syllables], note[7]= feature_type
    for note_idx, (num, length) in enumerate(notes):
        length = int(float(length))
        
        if word_pos >= len(syllables):
            break
        if num == 'rest':
            temp_lyrics.append([note_idx, '<None>', num, length])
        else:
            word_info = list(syll_with_word[word_pos])
            word_syl = word_info[0]
            word_idx = word_info[1]
            word = word_info[2]

            sylls = word.split('|')[-1].split('_')

            if word_idx in bb_idxs:
                temp_lyrics.append([note_idx, word_idx, num, length, word, word_syl, sylls, '<BB>'])
            elif word_idx in bl_idxs:
                temp_lyrics.append([note_idx, word_idx, num, length, word, word_syl, sylls, '<BL>'])
            else:
                temp_lyrics.append([note_idx, word_idx, num, length, word, word_syl, sylls, '<None>'])
            word_pos += 1

    last_note_idx = len(temp_lyrics) - 1
    for i, note in enumerate(temp_lyrics):
        print(note)
        if note[2] == 'rest':
            if i == 0 or i == last_note_idx:
                song['lyrics'].append([note[0], '<None>', note[2], note[3], '<None>', '<None>', '<None>', '<None>'])
            else:
                if temp_lyrics[i-1][1] == temp_lyrics[i+1][1]: # If rest in word
                    song['lyrics'].append(note[::] + [temp_lyrics[i+1][4]] + ['<None>'] + [temp_lyrics[i+1][-2::]])
                else:
                    song['lyrics'].append([note[0], '<None>', note[2], note[3], '<None>', '<None>', '<None>', '<None>'])
        else:
            song['lyrics'].append(note)
    
    song['lyrics'].append([last_note_idx + 1, '<None>', 'rest', '32', '<None>', '<None>', '<None>', '<None>'])
    corpus_file = open(output_dir + 'output.readable', 'w')
    for note in song["lyrics"]:
        corpus_file.write(json.dumps(note, ensure_ascii=False) + "\n")

def readable(generated, notes, checkpoint):
    # Save word and lyrics file
    # out_file = open(output_dir + 'output.txt', 'w')
    # lyrics_file = open(output_dir + 'lyrics.txt', 'w')

    line = []
    syllables = []
    syll_with_word = []
    word_idx = 0
    bb_idxs = set()
    bb_idxs.add(0)
    bl_idxs = set()

    for word in generated:
        if word == '<BB>|<null>':
            # lyrics_file.write(" ".join(w.split('|')[0] for w in line) + '\n\n')
            # out_file.write(" ".join(line) + '\n\n')
            line = []
            bb_idxs.add(word_idx)
        elif word == '<BL>|<null>':
            # lyrics_file.write(" ".join(w.split('|')[0] for w in line) + '\n')
            # out_file.write(" ".join(line) + '\n')
            line = []
            bl_idxs.add(word_idx)
        else:
            line.append(word)
            syllables += word.split('|')[-1].split('_')
            syll_with_word += [(s, word_idx, word) for s in word.split('|')[-1].split('_')]
            word_idx += 1

    # if len(line) > 0:
    #     lyrics_file.write(" ".join(w.split('|')[0] for w in line) + '\n')
    #     out_file.write(" ".join(line) + '\n')

    # Save note format file
    song = {'lyrics':[]}
    word_pos = 0
    temp_lyrics = []

    # note = [note_number, word_index, note_type, duration, word, syllable, feature_type]
    # note[0] = note_number, note[1] = word_index, note[2] = note_type(rest) or MIDI_number, note[3] = duration 
    # note[4] = word, note[5] = syllable, note[6] = [all syllables], note[7]= feature_type
    # print('Notes: ', notes)
    for note_idx, (num, length) in enumerate(notes):
        length = int(float(length))
        
        if word_pos >= len(syllables):
            break
        if num == 'rest':
            temp_lyrics.append([note_idx, '<None>', num, length])
        else:
            word_info = list(syll_with_word[word_pos])
            word_syl = word_info[0]
            word_idx = word_info[1]
            word = word_info[2]

            sylls = word.split('|')[-1].split('_')

            if word_idx in bb_idxs:
                temp_lyrics.append([note_idx, word_idx, num, length, word, word_syl, sylls, '<BB>'])
            elif word_idx in bl_idxs:
                temp_lyrics.append([note_idx, word_idx, num, length, word, word_syl, sylls, '<BL>'])
            else:
                temp_lyrics.append([note_idx, word_idx, num, length, word, word_syl, sylls, '<None>'])
            word_pos += 1

    last_note_idx = len(temp_lyrics) - 1
    for i, note in enumerate(temp_lyrics):
        if note[2] == 'rest':
            if i == 0 or i == last_note_idx:
                song['lyrics'].append([note[0], '<None>', note[2], note[3], '<None>', '<None>', '<None>', '<None>'])
            else:
                if temp_lyrics[i-1][1] == temp_lyrics[i+1][1]: # If rest in word
                    song['lyrics'].append(note[::] + [temp_lyrics[i+1][4]] + ['<None>'] + [temp_lyrics[i+1][-2::]])
                else:
                    song['lyrics'].append([note[0], '<None>', note[2], note[3], '<None>', '<None>', '<None>', '<None>'])
        else:
            song['lyrics'].append(note)
    
    song['lyrics'].append([last_note_idx + 1, '<None>', 'rest', '32', '<None>', '<None>', '<None>', '<None>'])
    # corpus_file = open(output_dir + 'output.readable', 'w')
    # for note in song["lyrics"]:
    #     corpus_file.write(json.dumps(note, ensure_ascii=False) + "\n")
    return song['lyrics']


def main(args):
    argparse_dict = vars(args)
    print("------ Parameters -----")
    for k, v in argparse_dict.items():
        print("{:>16}:  {}".format(k, v))

    notes = convert(args.midi)
    print("Notes: ", notes, type(notes))
    # save_notes = np.array(notes)
    np.save('c-LSTM-LM/test_output/notes.npy', notes)

    with torch.no_grad():
        generated_lyrics, positions, score = generate_deeper1(notes=notes, 
                                            param=args.param, checkpoint=args.checkpoint, 
                                            seed=args.seed, window=args.window, 
                                            temperature=args.temperature, LM_model = args.LM_model)

    np.save('c-LSTM-LM/test_output/genearated_', generated_lyrics)
    model_name = args.LM_model.split('.')[0]
    np.save('c-LSTM-LM/test_output/generated_' + model_name, generated_lyrics)

    # generated_lyrics = np.load('c-LSTM-LM/test_output/generated_Kopia av model_07.npy')
    save_lyrics(generated_lyrics, notes, args.output, args.checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """ Data parameter """
    parser.add_argument("-midi", "--midi", dest="midi", default="./c-LSTM-LM/sample_data/sample.midi", type=str, help="MIDI file")
    parser.add_argument("-output", "--output", dest="output", default="./c-LSTM-LM/test_output/", type=str, help="Output directory")

    """ Model parameter """
    checkpoint = 'checkpoint_03062020_1030' + '/' # Change checkpoint folder
    parser.add_argument("-param", "--param", dest="param", default="./c-LSTM-LM/" + checkpoint + "model.param.json", type=str, help="Parameter file path")
    parser.add_argument("-checkpoint", "--checkpoint", dest="checkpoint", default="./c-LSTM-LM/" + checkpoint, type=str, help="Checkpoint file path")

    """ Generation parameter """
    parser.add_argument("-seed", "--seed", dest="seed", default=0, type=int, help="Seed number for random library")
    parser.add_argument("-window", "--window", dest="window", default=20, type=int, help="Window size for beam search")
    parser.add_argument("-temperature", "--temperature", dest="temperature", default=3.0, type=float, help="Word sampling temperature")
    parser.add_argument("-LM_model", "--LM_model", dest="LM_model", default="model_06.pt", type=str, help="Model number of checkpoint") # Change model number
    args = parser.parse_args()
    main(args)