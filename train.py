# -*- coding: utf-8 -*-
import os
import sys
import argparse
import json
import random
import time
import numpy as np
import utils
import matplotlib.pyplot as plt
# from data import SongLyricDataset, collate_fn
from data2 import SongLyricDataset, collate_fn
from deeper_model import deepCLLM
from model import CLLM
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.nn.utils.rnn import pack_padded_sequence
from collections import defaultdict

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device type: %s"%device)

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

    """ Save dictionaries and models to checkpoint """
    with open(checkpoint + 'model.feature.json', 'w') as f:
        f.write(json.dumps(data_set.idx2feature))

    with open(checkpoint + 'model.vocab.json', 'w') as f:
        f.write(json.dumps(data_set.idx2word))

    with open(checkpoint + 'model.word2idx.json', 'w') as f:
        f.write(json.dumps(data_set.word2idx))

    with open(checkpoint + 'model.syllables.json', 'w') as f:
        f.write(json.dumps(data_set.word2syllables))

    with open(checkpoint + 'model.idx2syllable.json', 'w') as f:
        f.write(json.dumps(data_set.idx2syllable))


    with open(checkpoint + 'model.param.json', 'w') as f:
        f.write(json.dumps({"feature_idx_path": checkpoint+'model.feature.json',
                            "vocab_idx_path": checkpoint+'model.vocab.json',
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
    
    train_data_set, val_data_set = torch.utils.data.random_split(data_set, [train_size, validation_size])

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

    # test_data_loader = torch.utils.data.DataLoader(dataset=test_data_set,
    #                                               batch_size=batch_size,
    #                                               shuffle=True,
    #                                               num_workers=num_workers, 
    #                                               collate_fn=collate_fn)

    """ Load CLLM model """
    model = deepCLLM(word_dim=word_dim, melody_dim=melody_dim, syllable_size=data_syllable_size, word_size=data_word_size, feature_size=data_feature_size, num_layers=num_layers).to(device)

    # Calculate number of parameters in model
    utils.model_summary(model)

    pp=0
    for p in list(model.parameters()):
        nno=1
        for s in list(p.size()):
            nno = nno*s
        pp += nno
    print("Number of parameters: ", pp)


    """ Build Optimizers """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # lr = 0.001
    loss_criterion = nn.CrossEntropyLoss() # Combines LogSoftmax() and NLLLoss() (Negative log likelihood loss)

    """ Define traingin function """
    def train(epoch, data_set, data_loader):
        model.train() # Activate train mode

        # Log time
        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()
        sum_losses_syll = utils.AverageMeter()
        sum_losses_lyric = utils.AverageMeter()
        start_time = time.time()

        """ Batches """
        hidden = model.init_hidden(batch_size) # Creates a list of 3D layers with 1 x batch_size x hidden_dim

        for i, (syllable, lyric, melody, lengths) in enumerate(data_loader):
            # Take time
            elapsed = time.time()
            data_time.update((elapsed - start_time)*1000)
            
            # Check so that we got the correct batch size
            local_bs = lyric.size(0)
            if local_bs != args.batch_size:
                continue
            
            """ Move dataloaders to GPU """
            syllable = syllable.to(device)
            lyric = lyric.to(device)
            melody = melody.to(device).float()
            lengths = lengths.to(device)

            """ Remove first melody feature """
            melody = melody[:, 1:] # We dont really want to do this?

            """ Reset gradient to zero """
            optimizer.zero_grad()

            """ Detach hidden layers """
            hidden = utils.repackage_hidden(hidden) # Function from PyTorch NLP official example

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

            """ Backpropagation """
            loss = loss_syllable + loss_lyrics
            loss.backward()
            optimizer.step()

            """ Time """
            elapsed = time.time()
            batch_time.update((elapsed - start_time))

            """ Print progress """
            if i % log_interval == 0:
                lp.lprint('| Training Epoch: {:3d}/{:3d}  {:6d}/{:6d} '
                          '| lr:{:6.5f} '
                          '| {batch_time.avg:7.2f} s/batch '
                          '| {data_time.avg:5.2f} ms/data_load '
                          '| Loss(Syllable) {loss_s.avg:5.5f} '
                          '| Loss(Lyrics) {loss_l.avg:5.5f} |'
                          .format(epoch+1, num_epochs, i, len(data_loader), lr, 
                                  batch_time=batch_time,
                                  data_time=data_time, 
                                  loss_s=sum_losses_syll, 
                                  loss_l=sum_losses_lyric))
        return sum_losses_lyric.avg, sum_losses_syll.avg


    def validation(epoch, data_set, data_loader):
        model.eval()

        # Log time
        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()
        sum_losses_syll = utils.AverageMeter()
        sum_losses_lyric = utils.AverageMeter()
        start_time = time.time()

        """ Batches """
        hidden = model.init_hidden(batch_size) # Creates a list of 3D layers with 1 x batch_size x hidden_dim

        for i, (syllable, lyric, melody, lengths) in enumerate(data_loader):
            # Take time
            elapsed = time.time()
            data_time.update((elapsed - start_time)*1000)

            local_bs = lyric.size(0)
            if local_bs != args.batch_size:
                continue

            """ Move dataloaders to GPU """
            syllable = syllable.to(device)
            lyric = lyric.to(device)
            melody = melody.to(device).float()
            lengths = lengths.to(device)

            """ Remove first melody feature """
            melody = melody[:, 1:] # We dont really want to do this?

            """ Detach hidden layers """
            hidden = utils.repackage_hidden(hidden) # Function from PyTorch NLP official example

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

            """ Time """
            elapsed = time.time()
            batch_time.update((elapsed - start_time))

            """ Print progress """
            if i % log_interval == 0:
                lp.lprint('| Validation Epoch: {:3d}/{:3d}  {:6d}/{:6d} '
                          '| lr:{:6.5f} '
                          '| {batch_time.avg:7.2f} s/batch '
                          '| {data_time.avg:5.2f} ms/data_load '
                          '| Loss(Syllable) {loss_s.avg:5.5f} '
                          '| Loss(Lyrics) {loss_l.avg:5.5f} |'
                          .format(epoch+1, num_epochs, i, len(data_loader), lr, 
                                  batch_time=batch_time,
                                  data_time=data_time, 
                                  loss_s=sum_losses_syll, 
                                  loss_l=sum_losses_lyric))
        return sum_losses_lyric.avg, sum_losses_syll.avg


    def test(data_set, data_loader):
        print("test")

    def save_model(epoch):
        model.eval()
        with open(checkpoint+"model_%02d.pt"%(epoch+1), 'wb') as f:
            torch.save(model.state_dict(), f)

    """ Run Epochs """
    lp.lprint("------ Training -----", True)
    first_start_time = time.time()
    train_lyric_loss_vec = []
    train_syll_loss_vec = []
    val_lyric_loss_vec = []
    val_syll_loss_vec = []
    epoch_cnt = 0
    for epoch in range(num_epochs):
        # Training 
        train_lyric_loss, train_syll_loss = train(epoch, train_data_set, train_data_loader)
        train_lyric_loss_vec.append(train_lyric_loss)
        train_syll_loss_vec.append(train_syll_loss)
        lp.lprint("", True)

        # Validation
        with torch.no_grad():
            val_lyric_loss, val_syll_loss = validation(epoch, val_data_set, val_data_loader)
            val_lyric_loss_vec.append(val_lyric_loss)
            val_syll_loss_vec.append(val_syll_loss)
            lp.lprint("", True)

            # Save checkpoint
            if epoch_cnt%save_interval == 0:
                save_model(epoch)
        epoch_cnt += 1

        # Plot training loss
        plt.figure('train', (6, 3))
        plt.subplot(1, 2, 1)
        plt.title('Training lyric loss')
        plt.ylabel('Train lyric loss')
        plt.xlabel('Epoch')
        plt.plot(train_lyric_loss_vec)
        plt.subplot(1, 2, 2)
        plt.title('Training syllable loss')
        plt.ylabel('Train syllable loss')
        plt.xlabel('Epoch')
        plt.plot(train_syll_loss_vec)
        plt.show()
        plt.savefig('training_loss.png')

        # Plot validation loss
        plt.figure('validation', (6, 3))
        plt.subplot(1, 2, 1)
        plt.title('Validation lyric loss')
        plt.ylabel('Validation lyric loss')
        plt.xlabel('Epoch')
        plt.plot(val_lyric_loss_vec)
        plt.subplot(1, 2, 2)
        plt.title('Validation syllable loss')
        plt.ylabel('Validation syllable loss')
        plt.xlabel('Epoch')
        plt.plot(val_syll_loss_vec)
        plt.show()
        plt.savefig('validation_loss.png') 


        lp.lprint("-----------", True)
    elapsed = (time.time() - first_start_time)/60
    print("Elapsed = ", elapsed)
    lp.lprint('Total elapsed time: {elapsed:7.2f} minutes'.format(elapsed=elapsed))
    torch.cuda.empty_cache()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a conditional-LSTM language model to generate lyrics given melody")
    parser.add_argument('--settings_file', help='json file of settings, overrides everything else', type=str, default='settings.txt')
    parser.add_argument('-verbose', '--verbose', dest="verbose", default=1, type=int, help="verbose: 0 or 1")

    args = parser.parse_args()
    settings = vars(args)
    settings = utils.load_settings(settings)
    
    if not os.path.exists(settings['checkpoint']):
        os.mkdir(settings['checkpoint'])

    if args.verbose == 1:
        lp = utils.LogPrint(settings['checkpoint'] + 'model.log', True)
    else:
        lp = utils.LogPrint(settings['checkpoint'] + 'model.log', False)

    # Print settings
    lp.lprint("------ Parameters -----", True)
    for (k, v) in settings.items():
        lp.lprint("{:>16}:  {}".format(k, v), True)
    
    # Log settings
    with open(settings['checkpoint']+'args.json', 'w') as f:
        f.write(json.dumps(settings))
    
    # Update local variables
    locals().update(settings)

    # Redefine variables to avoid annoying text editor errors
    lr = lr
    batch_size = batch_size
    checkpoint = checkpoint
    word_size = word_size
    word_dim = word_dim
    melody_dim = melody_dim
    num_workers = num_workers
    seed = seed
    window = window
    train_rate = train_rate
    data = data
    num_epochs = num_epochs
    log_interval = log_interval
    save_interval = save_interval
    num_layers = num_layers
    


    main()
