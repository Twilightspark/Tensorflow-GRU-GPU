#-*- coding:utf-8 -*-
#author: Kai Zhang
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATABASE = {
    'type': 'csv',
    'name': 'data',
    'path': r'%s\database' % BASE_DIR,
    'folder': 'csv',
}

MODELBASE = {
    'type': 'tensorflow',
    'name': 'GRU',
    'path': 'database',
    'folder': 'model',
}

PARAMBASE = {
    'type': 'json',
    'name': 'parameter',
    'path': r'%s\database' % BASE_DIR,
    'folder': 'model',
}