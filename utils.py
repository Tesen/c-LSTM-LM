# -*- coding: utf-8 -*-
import json
import os
import numpy as np

def load_settings(settings):
    """
    Loading settings from the given json settings file. Overwrites command line input.
    """

    # Define settings path
    settings_path = './settings/' + settings['settings_file']
    print("Loading settings from: %s"%settings_path)

    settings_loaded = json.load(open(settings_path, 'r'))

    # Check for missing settings in file
    for key in settings.keys():
        if not key in settings_loaded:
            print(key, " not found in loaded settings")
    
    settings.update(settings_loaded)
    return settings
