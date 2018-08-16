#-*- coding:utf-8 -*-
#author: Kai Zhang
#function: GRU dynamic modeling project

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))#root directory
sys.path.append(BASE_DIR)

from core import main

if __name__ == '__main__':
    main.run()