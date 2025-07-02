'''A set of utilities for defining and working with namsel-ocr config files
'''

import json
import codecs
import os
import glob
import numpy as np
try:
    from .utils import create_unique_id, local_file
except ImportError:
    from utils import create_unique_id, local_file
from collections import Counter

import platform

if platform.system() == "Windows":
    CONF_DIR = r'.\confs'
else:
    CONF_DIR = './confs'

if not os.path.exists(CONF_DIR):
    os.mkdir(CONF_DIR)

def _open(fl, mode='r'):
    return codecs.open(fl, mode, encoding='utf-8')

default_config = {
    'page_type': 'book',
    'line_break_method': 'line_cut',
    'recognizer': 'hmm', # or probout
    'break_width': 2.0,
    'segmenter': 'stochastic', # or experimental
    'combine_hangoff': .6,
    'low_ink': False,
    'line_cluster_pos': 'top', # or center
    'viterbi_postprocessing': False, # determine if main is running using viterbi post processing
    'postprocess': False, # Run viterbi (or possibly some other) post processing
    'stop_line_cut': False,
    'detect_o': False,
    'clear_hr': False,
    'line_cut_inflation': 4, # The number of iterations when dilating text in line cut. Increase this value when need to blob things together
}

def update_default():
    json.dump(default_config, _open(os.path.join(CONF_DIR, 'default.conf'), 'w'), indent=1)

def create_misc_confs():
    from sklearn.grid_search import ParameterGrid
    params = {'break_width': [1.5, 2.0, 3.6, 5.0], 
              'recognizer': ['probout', 'hmm'], 'combine_hangoff': [.4, .6, .8], 
              'postprocess': [True, False], 'segmenter': ['experimental', 'stochastic'],
              'line_cluster_pos': ['top', 'center'],
              }
    grid = ParameterGrid(params)
    for pr in grid:
        Config(save_conf=True, **pr)


class Config(object):
    def __init__(self, path=None, save_conf=False, **kwargs):
        

        self.conf = default_config
        
            
        self.path = path
        if path:
            # Over-write defaults
            self._load_json_set_conf(path)
        
        
        # Set any manually specified config settings
        for k in kwargs:
            self.conf[k] = kwargs[k]
            
        if kwargs and save_conf:
            self._save_conf()
        
        
        # Set conf params as attributes to conf obj
        for k in self.conf:
            if k not in self.__dict__:
                setattr(self, k, self.conf[k])
        
        
    def _load_json_set_conf(self, path):
        try:
            conf = json.load(_open(path))
            for k in conf:
                self.conf[k] = conf[k]
        except IOError:
            print('Error in loading json file at %s. Using default config' % path)
            self.conf = default_config
    
     
    def _save_conf(self):
        '''Save a conf if it doesn't already exist'''
        
        confs = glob.glob(os.path.join(CONF_DIR, '*.conf'))

        for conf in confs:
            conf = json.load(_open(conf))
            if conf == self.conf:
                return
        else:
            json.dump(self.conf, _open(os.path.join(CONF_DIR, create_unique_id()+'.conf'), 'w'), indent=1)


