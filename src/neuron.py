#!/usr/bin/python3
r'''
* --- NEURON CLASS AND UNIT TESTS -----------------------------------------------------------------
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
except ModuleNotFoundError:
    import helpers


class Neuron:
    #? --- HOOK METHODS ---------------------------------------------------------------------------
    #? --------------------------------------------------------------------------------------------

    def __init__(self, config=None, n_inputs=None, f_act=None, q=None, q_scalar=None, bias=None):
        args = {key:val for key, val in locals().items() if key != 'self' and key != 'config'}
        self.__defaults = helpers.loadDefaults('neuron')
        self.__fatals = []
        self.__data = deepcopy(self.__defaults)

        tempConfig = {}
        for key, val in {key:val for key, val in args.items() if val != None}.items():
            tempConfig[key] = val
        if config is not None:
            for key, val in {key:val for key, val in config.items() if val != None}.items():
                tempConfig[key] = val
        self.setConfig(tempConfig)

    def __str__(self):
        ret = deepcopy(self.__data)
        ret['f_act'] = ret['f_act'].__name__[2:]
        return json.dumps(ret, indent=4)

    def __getitem__(self, key):
        return self.__data[key]


    #? --- HIDDEN METHODS -------------------------------------------------------------------------
    #? --------------------------------------------------------------------------------------------

    def __patchConfig(self, config):
        for key in self.__defaults.keys():
            if key not in config.keys():
                config[key] = None

        prev_config = deepcopy(self.__data)
        patched = deepcopy(self.__data)
        for key, val in {key:val for key, val in config.items() if val != None}.items():
            patched[key] = val

        if config['n_inputs'] is not None and config['q'] is None:
            patched['q'] = []
        elif config['n_inputs'] is None and config['q'] is not None:
            patched['n_inputs'] = -1
        elif config['n_inputs'] is None and config['q'] is None and config['q_scalar'] is not None:
            scalar_temp = patched['q_scalar'] / prev_config['q_scalar']
            patched['q'] = [scalar_temp * q for q in patched['q']]
        elif config['n_inputs'] is None and config['q'] is None and config['bias'] is not None:
            bias_temp = patched['bias'] - prev_config['bias']
            patched['q'] = [q + bias_temp for q in patched['q']]

        if patched['q'] == []:
            patched['q'] = self.__genRandQ(patched)
        if patched['n_inputs'] == -1:
            patched['n_inputs'] = len(patched['q'])

        return patched

    def __validateConfig(self, config):
        if len(config['q']) != config['n_inputs']:
            self.__fatals.append('{}-F-{} {}n_inputs{} and {}q length{} mismatch'.format(
                helpers.Format['RED'], helpers.Format['END'],
                helpers.Format['BOLD'], helpers.Format['END'],
                helpers.Format['BOLD'], helpers.Format['END'],
            ))
        if not callable(config['f_act']):
            self.__fatals.append('{}-F-{} {}{}{} is invalid selection for {}f_act{}'.format(
                helpers.Format['RED'], helpers.Format['END'],
                helpers.Format['BOLD'], config['f_act'], helpers.Format['END'],
                helpers.Format['BOLD'], helpers.Format['END']
            ))
        self.__qualify()

    def __qualify(self):
        if len(self.__fatals) > 0:
            print('', *self.__fatals, sep='\n')
            exit(len(self.__fatals))

    def __getActivationFunc(self, config):
        f_act = config['f_act'].lower()
        if f_act in self.actFunctions.keys():
            return self.actFunctions[f_act]

    def __genRandQ(self, config):
        n = config['n_inputs']
        q = np.random.randn(1, n)[0]
        q = config['q_scalar'] * q
        return list(q)

    def __normalizeQ(self, config):
        return [np.float64(q) for q in config['q']]


    #? --- ACTIVATION FUNCTIONS -------------------------------------------------------------------
    #? --------------------------------------------------------------------------------------------

    def __identity(self, data):
        self.__data['output'] = data

    def __step(self, data):
        self.__data['output'] = [1 if y > 0 else 0 for y in data]

    def __sigmoid(self, data):
        self.__data['output'] = [1/(1+np.exp(-y)) for y in data]

    def __tanh(self, data):
        self.__data['output'] = [(np.exp(y) - np.exp(-y))/(np.exp(y) + np.exp(-y)) for y in data]

    def __relu(self, data):
        self.__data['output'] = [y if y > 0 else 0 for y in data]

    actFunctions = {
        'identity': __identity,
        'step': __step,
        'sigmoid': __sigmoid,
        'tanh': __tanh,
        'relu': __relu
    }


    #? --- PUBLIC METHODS -------------------------------------------------------------------------
    #? --------------------------------------------------------------------------------------------

    def editConfig(self, n_inputs=None, f_act=None, q=None, q_scalar=None, bias=None):
        args = {key:val for key, val in locals().items() if key != 'self'}
        self.setConfig(args)

    def setConfig(self, config):
        if len(config.keys()) < len(self.__defaults.keys()):
            config = self.__patchConfig(config)
        if not callable(config['f_act']):
            config['f_act'] = self.__getActivationFunc(config)
        config['q'] = self.__normalizeQ(config)
        self.__validateConfig(config)
        self.__data = deepcopy(config)

    def forward(self, batch):
        if type(batch[0]) == float:
            batch = [batch]
        calc = [float(np.dot(self.__data['q'], sample) + self.__data['bias']) for sample in batch]
        self.__data['f_act'](self, calc)
        return self.output


    #? --- PROPERTIES -----------------------------------------------------------------------------
    #? --------------------------------------------------------------------------------------------

    @property
    def available_activation_functions(self):
        return list(self.actFunctions.keys())

    @property
    def config(self):
        return {key:val for key, val in self.__data.items() if key != 'output'}

    @property
    def output(self):
        return self.__data['output']
