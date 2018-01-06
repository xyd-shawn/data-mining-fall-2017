# -*- coding: utf-8 -*-

import os
import sys

def sort_index(x):
    idx = [[i, v] for i, v in enumerate(x)]
    idx = sorted(idx, key=lambda t: t[1], reverse=True)
    idx = [i for i, v in idx]
    return idx
