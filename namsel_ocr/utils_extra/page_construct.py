import os
import gzip
import pickle as pickle
import shelve

allchars = shelve.open('/home/zr/letters/allchars_dict2')

TSEK_APPR_MEAN = 19.5 # Fudge -- for use with Nyinggyud in particular.

def load_pkl_page(pkl_path):
    if os.path.exists(pkl_path):
        fl = gzip.open(pkl_path)
        return pickle.Unpickler(fl).load()

def construct_content(info):
    content = []
    for i, char in enumerate(info[:-1]):
        content.append(allchars['label_chars'][char[2]])
        if info[i+1][0] == char[0] +1:
            content.append('\n')
        elif info[i+1][3] - char[4] >= TSEK_APPR_MEAN *2:
            content.append(' ')
    content.append(allchars['label_chars'][info[-1][2]])
    return ''.join(content)
