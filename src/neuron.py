#!/usr/bin/python3
r'''
# Neuron Class

---

Responsible for being awesome
'''

import numpy as np
import json
from copy import deepcopy
try:
    import src.helpers as helpers
except ModuleNotFoundError:
    import helpers
    import unittest


class Neuron:
    #? --- HOOK METHODS ---------------------------------------------------------------------------
    #? --------------------------------------------------------------------------------------------

    def __init__(self, config=None, n_inputs=None, f_act=None, q=None, q_scalar=None, bias=None):
        args = {key:val for key, val in locals().items() if key != 'self'}
        self.__defaults = helpers.loadDefaults('neuron')
        self.__fatals = []
        cfg = args.pop('config')
        if cfg is not None:
            self.setConfig(cfg)
        else:
            self.__data = deepcopy(self.__defaults)
        self.editConfig(**args)
        self.__update()

    def __str__(self):
        ret = deepcopy(self.__data)
        for key, val in ret.items():
            if callable(val):
                ret[key] = f'{val.__name__}()'
        return json.dumps(ret, indent=4)

    def __getitem__(self, key):
        return self.__data[key]


    #? --- HIDDEN METHODS -------------------------------------------------------------------------
    #? --------------------------------------------------------------------------------------------

    def __update(self):
        for key in self.__defaults.keys():
            if key not in self.__data.keys():
                self.__data[key] = None
        for key, val in self.__data.items():
            if val is None:
                self.__data[key] = self.__defaults[key]
        if type(self.__data['f_act']) == str:
            self.__data['f_act'] = self.__getActivationFunc()
        if self.__data['q'] == []:
            self.__data['q'] = self.__genRandQ()
        else:
            self.__data['q'] = self.__normalizeQ()
        self.__qualify()

    def __qualify(self):
        if len(self.__fatals) > 0:
            print('\n'.join(self.__fatals))
            exit(1)

    def __getActivationFunc(self):
        f_act = self.__data['f_act'].lower()
        if f_act in self.actFunctions.keys():
            return self.actFunctions[f_act]
        self.__fatals.append('{}-F-{} {}{}{} is an invalid selection for {}f_act{}'.format(
            helpers.Format['RED'], helpers.Format['END'],
            helpers.Format['BOLD'], f_act, helpers.Format['END'],
            helpers.Format['BOLD'], helpers.Format['END'],
        ))

    def __genRandQ(self):
        n = self.__data['n_inputs']
        if type(n) == int:
            q = np.random.randn(1, n)[0]
            q = self.__data['q_scalar'] * q
            return list(q)
        self.__fatals.append('{}-F-{} {}{}{} is an invalid value for {}n_inputs{}'.format(
            helpers.Format['RED'], helpers.Format['END'],
            helpers.Format['BOLD'], n, helpers.Format['END'],
            helpers.Format['BOLD'], helpers.Format['END'],
        ))

    def __normalizeQ(self) -> list:
        types = list(set([type(q) for q in self.__data['q']]))
        if len(types) > 1 or (types[0] != int and types[0] != np.float64):
            self.__fatals.append('{}-F-{} {}q{} has invalid type(s); types found: {}'.format(
                helpers.Format['RED'], helpers.Format['END'],
                helpers.Format['BOLD'], helpers.Format['END'],
                types
            ))
        if types[0] == int or types[0] == float:
            return [np.float64(q) for q in self.__data['q']]
        return self.__data['q']


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
        for key, val in args.items():
            if key not in self.__data.keys():
                self.__data[key] = None
            if val is not None:
                self.__data[key] = val
        self.__update()

    def setConfig(self, config):
        self.__data = deepcopy(config)
        self.__update()

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


#! --- UNIT TEST ----------------------------------------------------------------------------------
#! ------------------------------------------------------------------------------------------------

class TestNeuron(unittest.TestCase):
    def setUp(self):
        self.__n = 2
        self.__neuron = Neuron(n_inputs=self.__n)
        self.__defaults = helpers.loadDefaults('neuron')
        self.__batch = [
            [-1, -1], [-1, 0], [-1, 1],
            [ 0, -1], [ 0, 0], [ 0, 1],
            [ 1, -1], [ 1, 0], [ 1, 1]
        ]

    def __configsEqual(self, config1, config2):
        special_cases = ['f_act']
        for cfg1, cfg2 in zip(config1.items(), config2.items()):
            key, val1 = cfg1
            _, val2 = cfg2
            if key not in special_cases:
                self.assertEqual(val1, val2)
            elif key == 'f_act':
                self.assertEqual(val1.lower(), val2.__name__[2:])

    def test_01_defaults(self):
        special_cases = ['n_inputs', 'f_act', 'q']
        for key, val in self.__defaults.items():
            if key not in special_cases:
                self.assertEqual(val, self.__neuron[key])
            elif key == 'f_act':
                self.assertEqual(val.lower(), self.__neuron[key].__name__[2:])
            elif key == 'n_inputs':
                self.assertEqual(self.__n, self.__neuron[key])

    def test_02_q(self):
        self.assertEqual(self.__n, len(self.__neuron['q']))

    def test_03_edit(self):
        key = 'q_scalar'
        self.assertEqual(self.__defaults[key], self.__neuron[key])
        new_q_scalar = self.__defaults[key]*2
        self.__neuron.editConfig(q_scalar=new_q_scalar)
        self.assertEqual(new_q_scalar, self.__neuron[key])

    def test_04_full_config(self):
        config = {
            'n_inputs': 5,
            'f_act': 'identity',
            'q': [1, 2, 3, 4, 5],
            'q_scalar': -1,
            'bias': 3.14,
            'output': []
        }
        self.__neuron.setConfig(config)
        self.__configsEqual(config, self.__neuron.config)

    def test_05_sparce_config(self):
        config = {
            'n_inputs': 5,
            'q': [np.float64(q) for q in [1, 2, 3, 4, 5]]
        }
        self.__neuron.setConfig(config)
        for key, val in self.__defaults.items():
            if key not in config.keys():
                config[key] = val
        self.__configsEqual(config, self.__neuron.config)

    def test_06_identity_af(self):
        self.__neuron.editConfig(f_act='identity', q=[1 for _ in range(self.__n)], bias=0)
        self.__neuron.forward(self.__batch)
        self.assertEqual([-2.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 2.0], self.__neuron.output)

    def test_07_step_af(self):
        self.__neuron.editConfig(f_act='step', q=[1 for _ in range(self.__n)], bias=0)
        self.__neuron.forward(self.__batch)
        self.assertEqual([0, 0, 0, 0, 0, 1, 0, 1, 1], self.__neuron.output)

    def test_08_relu(self):
        self.__neuron.editConfig(f_act='relu', q=[1 for _ in range(self.__n)], bias=0)
        self.__neuron.forward(self.__batch)
        self.assertEqual([0, 0, 0, 0, 0, 1.0, 0, 1.0, 2.0], self.__neuron.output)

    def test_09_sigmoid_af(self):
        self.__neuron.editConfig(f_act='sigmoid', q=[1 for _ in range(self.__n)], bias=0)
        self.__neuron.forward(self.__batch)
        self.assertEqual([0.11920292202211755, 0.2689414213699951, 0.5, 0.2689414213699951, 0.5, 0.7310585786300049, 0.5, 0.7310585786300049, 0.8807970779778823], self.__neuron.output)

    def test_10_tanh_af(self):
        self.__neuron.editConfig(f_act='tanh', q=[1 for _ in range(self.__n)], bias=0)
        self.__neuron.forward(self.__batch)
        self.assertEqual([-0.964027580075817, -0.7615941559557649, 0.0, -0.7615941559557649, 0.0, 0.7615941559557649, 0.0, 0.7615941559557649, 0.964027580075817], self.__neuron.output)


if __name__ == '__main__':
    print('{}-I-{} Running unit tests on {}Neuron{} class...'.format(
        helpers.Format['GREEN'], helpers.Format['END'],
        helpers.Format['BOLD'], helpers.Format['END']
    ))
    print('----------------------------------------------------------------------')
    unittest.main(verbosity=2)

