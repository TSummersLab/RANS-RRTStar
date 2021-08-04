#!/usr/bin/env python3
"""
File searching functions
"""

import os
from os.path import splitext
import numpy as np


def get_all_filenames(dir):
    return [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]


def get_all_directories(dir):
    return [f for f in os.listdir(dir) if not os.path.isfile(os.path.join(dir, f))]


def get_timestr(dir=None, method='last', filename=None):
    if filename is not None:
        base, ext = splitext(filename)
        if ext == '':
            words = base.split('_')
            for word in words:
                if len(word) == 10 and word.isnumeric():
                    return word
    elif method == 'last':
        tmax = 0
        for filename in get_all_filenames(dir):
            base, ext = splitext(filename)
            if ext == '':
                words = base.split('_')
                for word in words:
                    if len(word) == 10 and word.isnumeric():
                        t = int(word)
                        if t > tmax:
                            tmax = t
        return str(tmax)
    else:
        raise ValueError('Invalid filename selection method!')
    return ''


def get_filename(dir, tstr='last', prefix='', suffix=''):
    if not tstr.isnumeric():
        tstr = get_timestr(dir, tstr)
    for filename in get_all_filenames(dir):
        base, ext = splitext(filename)
        if base.startswith(prefix) and base.endswith(suffix):
            words = base.split('_')
            if tstr in words:
                return filename
    return None


def get_vals_from_dirnames(dir, tstr, pos=0):
    # pos: int, the position in the word list of the desired value
    str_list = []
    val_list = []
    for dirname in get_all_directories(dir):
        words = dirname.split('_')
        if tstr in words:
            w = words[pos]
            str_list.append(w)
            val_list.append(float(w))
    idxs = np.argsort(val_list)

    def reorder_list(a, idxs):
        return np.array(a)[idxs].tolist()

    return reorder_list(str_list, idxs), reorder_list(val_list, idxs)
