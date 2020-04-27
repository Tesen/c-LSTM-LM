# -*- coding: utf-8 -*-
import json
import os
import sys
import numpy as np
import torch

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count


class LogPrint:
    def __init__(self, file_path, err):
        self.file = open(file_path, "w", buffering=1)
        self.err = err

    def lprint(self, text, ret=False, ret2=False):
        if self.err:
            if ret == True:
                if ret2 == True:
                    sys.stderr.write("\n" + text + "\n")
                else:
                    sys.stderr.write("\r" + text + "\n")
            else:
                sys.stderr.write("\r" + text)
        self.file.write(text + "\n")


def load_settings(settings):
    """
    Loading settings from the given json settings file. Overwrites command line input.
    """

    # Define settings path
    settings_path = './settings/' + settings['settings_file']
    print("Loading settings from: %s"%settings_path)

    settings_loaded = json.load(open(settings_path, 'r'))

    # Check for missing settings in file
    # for key in settings.keys():
    #     if not key in settings_loaded:
    #         print(key, " not found in loaded settings")
    
    settings.update(settings_loaded)
    return settings

# Function from PyTorch NLP official example
def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)