#!/usr/bin/python3

from time import time
from pathlib import Path
import json


#? --- CLASSES ------------------------------------------------------------------------------------
#? ------------------------------------------------------------------------------------------------

class Timer:
    def __init__(self, name=''):
        self.__name = name

    def __enter__(self):
        self.__tstart = time()

    def __exit__(self, type, value, tb):
        t = time() - self.__tstart
        print(f'{Format.GREEN}{self.__name}{Format.END} elapsed time: {t:.3f} second(s)')


#? --- METHODS ------------------------------------------------------------------------------------
#? ------------------------------------------------------------------------------------------------

def loadDefaults(key):
    defaults_path = Path(__file__).resolve().parent / 'defaults.json'
    with open(defaults_path, 'r') as f:
        data = json.load(f)[key]
    return data


#? --- VARIABLES ----------------------------------------------------------------------------------
#? ------------------------------------------------------------------------------------------------

Format = {
    'GREEN': '\033[92m',
    'RED': '\033[91m',
    'BOLD': '\033[1m',
    'END': '\033[0m'
}
