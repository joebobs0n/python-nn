#!/usr/bin/python3
r'''
* --- LAYER CLASS (NEURAL NETWORK INTERMEDIATE LEVEL) ---------------------------------------------
* -------------------------------------------------------------------------------------------------
*
*
* Author: joebobs0n
* Last Edited: 25 Jun 2021
* -------------------------------------------------------------------------------------------------
'''

from copy import deepcopy
import numpy as np
import json
try:
    import src.helpers as helpers
    from src.neuron import Neuron
except ModuleNotFoundError:
    import helpers
    from neuron import Neuron


class Layer:
    #? --- HOOK METHODS ---------------------------------------------------------------------------
    #? --------------------------------------------------------------------------------------------

    def __init__(self, config=None, n_neurons=None):
        self.__defaults = helpers.loadDefaults('layer')
        args = {key:val for key, val in locals().items() if key != 'self'}
        cfg = args.pop('config')
        if cfg is not None:
            self.__data = cfg
        else:
            self.__data = self.__defaults


    #? --- HIDDEN METHODS -------------------------------------------------------------------------
    #? --------------------------------------------------------------------------------------------


    #? --- PUBLIC METHODS -------------------------------------------------------------------------
    #? --------------------------------------------------------------------------------------------

    def editConfig(self):
        pass

    def setConfig(self, config):
        pass

    def forward(self):
        pass


    #? --- PROPERTIES -----------------------------------------------------------------------------
    #? --------------------------------------------------------------------------------------------

