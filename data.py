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
lengths = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 16, 32]

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

        
        print(features)
        # NOTE: Do I need to add rest and note types to the dataset? In that case, I must augment the dataset even further

        # Create index dictionaries for features
        sorted_features = sorted(features)
        self.feature2idx = dict((f, i) for i, f in enumerate(sorted_features))
        self.idx2feature = dict((i, f) for i, f in enumerate(sorted_features))
        self.feature_size = len(self.feature2idx)

        print(self.feature2idx)
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
                print("word " + str(note[1]) + ": ", note) # Print note for understanding
                # note = [note_number, word_number, note_type, duration, word, syllable, feature_type]
                # note[0] = note_number 
                # note[1] = word_number 
                # note[2] = note_type(rest) or MIDI_number 
                # note[4] = duration 
                # note[5] = word 
                
                # note[6] = syllable
                # note[7] = [all syllables] 
                # note[8]= feature_type
                word_idx = note[1]
                if word_idx != old_word_idx:
                    word_lower = note[5].lower()
                    word_dict[word_lower] += 1
                    syll_dict[word_lower] += len(note[7])
                    print("number of syllalbes = %s"%len(note[7]))


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
            # Add number of syllables for each word
            syllables.add(np.round(syll_dict[word]/freq))

            # Build word/index dictionaries
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            idx += 1

        self.word_size = len(self.word2idx)
        self.syllable_size = max(syllables) + 10

        print("word size: ", self.word_size)
        print("syllable size: ", self.syllable_size)


        # Create syllable, lyric and melody embeddings
        self.idx2lyrics = []
        self.idx2syllable = []
        self.idx2melody = []

        # For each song
        for file in files:
            notes = np.load(os.path.join(data, file), allow_pickle=True)
            print(notes)

            # Define starting state
            old_word_idx = "<None>"
            tag_stack = ["<WORD>"]
            
            # Initiate arrays
            syllables = []
            lyrics = []
            melody = []
            
            # For each word in song lyric increment word occurance dictionary
            
            # NOTE: A feature vectore "feature[]" contains the indexes of the features previous tag (BB,BL or WORD) 
            # Then it also contain the 10 previous note or rest indexes based on position in window as well as the specific rests/notes indexed duration
            # Then it also contain the 10 upcoming note or rests indexes and durations in the same manner.
            # So to conclude it looks like this: feature = [prev_tag, [previous notes/rests], [upcoming notes/rest]]
            # eg.: index of the following [prev_tag = BL, note[-10]=note, note_dur, note[-9]=rest, rest_dur, ...,
            #                                note[-1]=note, note_dur, note[0]= note,note_dur, note[1]=rest, rest_dur, note[2] = note, note_dur, note[3]=note, note_dur, ...]
            # which in this case would be something like: [280, 240, 108, ...]



            for i, word in enumerate(song_features[2]):
                feauture_type = song_features[4][0][i][0]
                print("feature type: ", feauture_type)
                # This defines and lists the window for previous and upcoming notes
                prev_i = i - window + 1
                if prev_i < 0: 
                    prev_i = 0

                next_i = i + window
                if next_i > len(song_features[0]):
                    next_i = len(song_features[0])

                # If feature type is BB
                if feauture_type == "<BB>":
                    feature = [] # Initiatie the feature vector which is to contain the 

                    w_idx = self.word2idx.get("<BB>|<null>") # Get word index of BB feature
                    lyrics.append(w_idx)
                    syllables.append(1)

                    prev_tag = self.feature2idx["prev_tag=%s"%tag_stack[-1]]
                    feature.append(prev_tag)

                    
                    for j, jj in enumerate(range(prev_i, i)): # j = [0:8], jj = [start_of_window_note:end_of_window_note]
                        rest_num = song_features[1][jj][0][2]
                        note_num = song_features[1][jj][1]

                        # print("i = %s, j = %s, jj = %s, prev_i = %s, i-prev_i-j = %s"%(i, j, jj, prev_i, i-prev_i-j))
                        if rest_num > 0:
                            note_num = self.feature2idx["note[-%s]=rest"%(i-prev_i-j)]
                            note_duration = self.feature2idx["length[-%s]=%s",(i-prev_i-j, rest_num)]
                        else:
                            note_num = self.feature2idx["note[-%s]=note"%(i-prev_i-j)]
                        

                              


                    tag_stack.append("<BB>")

                    # for j in range(prev_i, i+1):
                        # print("hej")




        self.feature_size = 1
        








    def __len__(self):
        return len(self.idx2lyrics)

    def __getitem__(self, idx):
        sample = 0
        return sample

