# -*- coding: utf-8 -*-
import torch
from torch import nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

class CLLM(nn.Module):
    def __init__(self, word_dim, melody_dim, syllable_size, word_size, feature_size, num_layers):
        super(CLLM,  self).__init__()
        self.hidden_dim = word_dim + melody_dim
        
        """ Word embedding """
        self.embedding = nn.Embedding(word_size, word_dim)

        """ Melody vector """
        self.fc_melody = nn.Linear(feature_size, melody_dim) # Fully connected layer

        """ LSTM """
        self.rnn = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=num_layers, bias=True, batch_first=True, bidirectional=False)

        """ Output """
        self.fc_lyrics_out = nn.Linear((self.hidden_dim), word_size) # Fully connected layer
        self.fc_syllables_out = nn.Linear(self.hidden_dim, int(syllable_size)) # Fully connected layer
        
        """ Util """
        self.relu = nn.ReLU(True)
        self.bn_lyrics = nn.BatchNorm1d(word_size)
        self.bn_syllables = nn.BatchNorm1d(syllable_size)

    def forward(self, lyrics, melody, lengths, hidden):
        lengths = lengths - 1
        local_batch_size = lyrics.shape[0]

        """ Word embedding """
        word_emb = self.embedding(lyrics)

        """ Melody vector """
        melody_vec = self.relu(self.fc_melody(melody))

        """ Input vector """
        input_vec = torch.cat((word_emb, melody_vec), dim=2)
        input_vec = pack_padded_sequence(input_vec, lengths, batch_first=True)

        """ RNN """
        output, hidden = self.rnn(input_vec, hidden)

        """ Output """
        lyrics_output = self.fc_lyrics_out(output[0])
        syllable_output = self.fc_syllables_out(output[0])

        if local_batch_size > 1:
            lyrics_output = self.bn_lyrics(lyrics_output) # Batch normalization
            syllable_output = self.bn_syllables(syllable_output)

        return syllable_output, lyrics_output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(1, bsz, self.hidden_dim), weight.new_zeros(1, bsz, self.hidden_dim))

