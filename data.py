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
note_types = ['note', 'rest']
tags = ['<WORD>', '<BB>', '<BL>']
lengths = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0, 32.0]

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

class SongLyricDataset(data.Dataset):
    def __init__(self, data, word_size, window):
        """ Create feature vocab and index dictionaries """
        # Create feature vocabulary
        for i in range(window):
            for note_type in note_types:
                features.append('note[%s]=%s'%(i, note_type))
                features.append('note[%s]=%s'%(-(i+1), note_type))
            for length in lengths:
                features.append('length[%s]=%s'%(i, length))
                features.append('length[%s]=%s'%(-(i+1), length))
        for tag in tags:
            features.append('prev_tag=%s'%tag)

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
            notes = np.load(os.path.join(data, file), allow_pickle=True)
            old_word_idx = "<None>"
            # For each word in song lyric increment word occurance dictionary
            for note in notes:
                # note = [note_number, word_index, note_type, duration, word, syllable, feature_type]
                # note[0] = note_number, note[1] = word_index, note[2] = note_type(rest) or MIDI_number, note[3] = duration 
                # note[4] = word, note[5] = syllable, note[6] = [all syllables], note[7]= feature_type
                word_idx = note[1]
                if word_idx != old_word_idx:
                    word_lower = note[5].lower()
                    word_dict[word_lower] += 1
                    syll_dict[word_lower] += len(note[6])


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
        print("Number of unique words: %s" %len(word_dict.items()))
        for word, freq in sorted(word_dict.items(), key=lambda x:x[1], reverse=True)[:word_size:]: # Sort word_dict after frequency and limit size to word_size (size of our dictionary)
            # Add number of syllables for each word, calculate average number of syllables per word and average
            syllables.add(np.round(syll_dict[word]/freq))

            # Build word/index dictionaries
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            idx += 1

        self.word_size = len(self.word2idx)
        self.syllable_size = int(max(syllables) + 10)

        # print("word size: ", self.word_size)
        # print("syllable size: ", self.syllable_size)


        """ Create syllable, lyric and melody embeddings """
        self.idx2lyrics = []
        self.idx2syllable = []
        self.idx2melody = []

        # For each song
        for file in files:
            notes = np.load(os.path.join(data, file), allow_pickle=True)

            # Define starting state
            old_word_idx = "<None>"
            tag_stack = ["<WORD>"]
            
            # Initiate arrays
            syllables = []
            lyrics = []
            melody = []

            old_word_idx = "<None>"
            
            """ For each word in song lyric increment word occurance dictionary """
            
            # NOTE: A feature vectore "feature[]" contains the indexes of the features previous tag (BB,BL or WORD) 
            # Then it also contain the 10 previous note or rest indexes based on position in window as well as the specific rests/notes indexed duration
            # Then it also contain the 10 upcoming note or rests indexes and durations in the same manner.
            # So to conclude it looks like this: feature = [prev_tag, [previous notes/rests], [upcoming notes/rest]]
            # eg.: index of the following [prev_tag = BL, note[-10]=note, note_dur, note[-9]=rest, rest_dur, ...,
            #                                note[-1]=note, note_dur, note[0]= note,note_dur, note[1]=rest, rest_dur, note[2] = note, note_dur, note[3]=note, note_dur, ...]
            # which in this case would be something like: [280, 240, 108, ...]

            for i, note in enumerate(notes):
                feauture_type = note[7]

                word_idx = note[1]
                if word_idx != old_word_idx:
                    # This defines and lists the window for previous and upcoming notes
                    prev_i = i - window + 1
                    if prev_i < 0: 
                        prev_i = 0
                    prev_notes = notes[prev_i:i]

                    next_i = i + window
                    if next_i > len(notes):
                        next_i = len(notes)
                    next_notes = notes[i:next_i]

                    # If feature type is BB
                    if feauture_type == "<BB>":
                        feature = [] # Initiatie the feature vector which is to contain the 

                        w_idx = self.word2idx.get("<BB>|<null>") # Get word index of BB feature
                        lyrics.append(w_idx) # Append lyric array with feature
                        syllables.append(1) # Append syllable array with 1

                        prev_tag = self.feature2idx["prev_tag=%s"%tag_stack[-1]]
                        feature.append(prev_tag)

                        # For previous 8 notes in window
                        for j, prev_note in enumerate(prev_notes):
                            if prev_note[2] == 'rest':
                                note_num = self.feature2idx["note[-%s]=rest"%(len(prev_notes)-j)]
                            else:
                                note_num = self.feature2idx["note[-%s]=note"%(len(prev_notes)-j)]
                            
                            note_duration = self.feature2idx["length[-%s]=%s"%((len(prev_notes)-j), get_correct_length(prev_note[3]))]
                            
                            feature.append(note_num)
                            feature.append(note_duration)


                        # For upcoming 8 notes in the window
                        for j, next_note in enumerate(next_notes):
                            if next_note[2] == 'rest':
                                note_num = self.feature2idx["note[%s]=rest"%(len(next_note)-j)]
                            else:
                                note_num = self.feature2idx["note[%s]=note"%(len(next_note)-j)]

                            note_duration = self.feature2idx["length[%s]=%s"%(len(next_note)-j, get_correct_length(next_note[3]))]

                            feature.append(note_num)
                            feature.append(note_duration)

                        # Pad feature vector (add elements to fill the array)
                        feature = [feature[0]]*(39 - len(feature)) + feature # (adds the first element several times if its shorter than 39)

                        # The feature vector is built up as indexes of ['prev_tag', 'note_num', 'note_duration', 'note_num', 'note_duration', ..., 'next_tag', 'note_num', 'note_duration, ...]
                        melody.append(feature[::])
                        tag_stack.append("<BB>")

                    if feauture_type == "<BL>":
                        feature = [] # Initiatie the feature vector which is to contain the 

                        w_idx = self.word2idx.get("<BL>|<null>") # Get word index of BB feature
                        lyrics.append(w_idx) # Append lyric array with feature
                        syllables.append(1) # Append syllable array with 1

                        prev_tag = self.feature2idx["prev_tag=%s"%tag_stack[-1]]
                        feature.append(prev_tag)

                        # For previous 8 notes in window
                        for j, prev_note in enumerate(prev_notes):
                            if prev_note[2] == 'rest':
                                note_num = self.feature2idx["note[-%s]=rest"%(len(prev_notes)-j)]
                            else:
                                note_num = self.feature2idx["note[-%s]=note"%(len(prev_notes)-j)]
                            
                            note_duration = self.feature2idx["length[-%s]=%s"%((len(prev_notes)-j), get_correct_length(prev_note[3]))]
                            
                            feature.append(note_num)
                            feature.append(note_duration)


                        # For upcoming 8 notes in the window
                        for j, next_note in enumerate(next_notes):
                            if next_note[2] == 'rest':
                                note_num = self.feature2idx["note[%s]=rest"%(len(next_note)-j)]
                            else:
                                note_num = self.feature2idx["note[%s]=note"%(len(next_note)-j)]

                            note_duration = self.feature2idx["length[%s]=%s"%(len(next_note)-j, get_correct_length(next_note[3]))]

                            feature.append(note_num)
                            feature.append(note_duration)

                        # Pad feature vector (add elements to fill the array)
                        feature = [feature[0]]*(39 - len(feature)) + feature # (adds the first element several times if its shorter than 39)

                        # The feature vector is built up as indexes of ['prev_tag', 'note_num', 'note_duration', 'note_num', 'note_duration', ..., 'next_tag', 'note_num', 'note_duration, ...]
                        melody.append(feature[::])
                        tag_stack.append("<BL>")                        

                    feature = []
                    w_idx = self.word2idx.get(note[4], self.word2idx["<unk>"])
                    lyrics.append(w_idx) # Append lyric array with feature index
                    syllables.append(len(note[6])) # Append sylable array with number of features

                    prev_tag = self.feature2idx["prev_tag=%s"%tag_stack[-1]]
                    feature.append(prev_tag)

                    # For previous 8 notes in window
                    for j, prev_note in enumerate(prev_notes):
                        if prev_note[2] == 'rest':
                            note_num = self.feature2idx["note[-%s]=rest"%(len(prev_notes)-j)]
                        else:
                            note_num = self.feature2idx["note[-%s]=note"%(len(prev_notes)-j)]
                        
                        note_duration = self.feature2idx["length[-%s]=%s"%((len(prev_notes)-j), get_correct_length(prev_note[3]))]
                        
                        feature.append(note_num)
                        feature.append(note_duration)

                    # For upcoming 8 notes in the window
                    for j, next_note in enumerate(next_notes):
                        if next_note[2] == 'rest':
                            note_num = self.feature2idx["note[%s]=rest"%(len(next_note)-j)]
                        else:
                            note_num = self.feature2idx["note[%s]=note"%(len(next_note)-j)]

                        note_duration = self.feature2idx["length[%s]=%s"%(len(next_note)-j, get_correct_length(next_note[3]))]

                        feature.append(note_num)
                        feature.append(note_duration)

                    # Pad feature vector (add elements to fill the array)
                    feature = [feature[0]]*(39 - len(feature)) + feature # (adds the first element several times if its shorter than 39)
                    melody.append(feature[::])
                    tag_stack.append("<WORD>")

            old_word_idx = word_idx
            
            # Append syllable, lyric and melody object array with arrays
            self.idx2syllable.append(syllables[::])
            self.idx2lyrics.append(lyrics[::])
            self.idx2melody.append(lyrics[::])

    def __len__(self):
        return len(self.idx2lyrics)

    def __getitem__(self, idx):
        syllables = torch.Tensor(self.idx2syllable[idx])
        lyrics = torch.Tensor(self.idx2lyrics[idx])
        melody = self.idx2melody[idx]

        return syllables, lyrics, melody. self.feature_size


def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    _syllables, _lyrics, _melody, feature_size = zip(*data)
    
    lengths = [len(_lyric) for _lyric in _lyrics] # Creates an array of the lengths of each songs lyrics
    max_length = lengths[0]
    
    lyrics = torch.zeros(len(_lyrics), max_length).long() # Initialise tensors
    syllables = torch.zeros(len(_syllables), max_length).long() # Initialise tensors
    melody = torch.zeros(len(_melody), max_length, feature_size[0]).long() # Initialise tensors

    for i, _lyric in enumerate(_lyrics):
        end = lengths[i]
        lyrics[i, :end] = _lyric[:end] # Create one long tensor for all songs
        syllables[i, :end] = _syllables[i][:end] # Create one long tensor for all songs
        melody[i, :end].scatter_(1, torch.Tensor(_melody[i]).long(), 1)

    lengths = torch.Tensor(lengths).long()

    return syllables, lyrics, melody, lengths


