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
        special_cases = ['f_act', 'q']
        for tempConfig1, tempConfig2 in zip(config1.items(), config2.items()):
            key, val1 = tempConfig1
            _, val2 = tempConfig2
            if key not in special_cases:
                self.assertEqual(val1, val2)
            elif key == 'f_act':
                f1 = val1.lower() if type(val1) == str else val1.__name__[2:]
                f2 = val2.lower() if type(val2) == str else val2.__name__[2:]
                self.assertEqual(f1, f2)
            elif key == 'q':
                self.assertEqual(len(val1), len(val2))

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
        configs = [
            {'n_inputs': 5},
            {'f_act': 'tanh'},
            {'q': [0.3, -1.2]},
            {'q_scalar': -1},
            {'bias': 1}
        ]
        for c in configs:
            modified = deepcopy(self.__neuron)
            self.__configsEqual(self.__neuron.config, modified.config)
            modified.setConfig(c)

    def test_06_q_scalar(self):
        modified = deepcopy(self.__neuron)
        modified.editConfig(q_scalar=self.__neuron['q_scalar']/2)
        self.assertEqual([q/2 for q in self.__neuron['q']], modified['q'])
        modified = deepcopy(self.__neuron)
        modified.editConfig(q_scalar=self.__neuron['q_scalar']*-1)
        self.assertEqual([q*-1 for q in self.__neuron['q']], modified['q'])

    def test_07_bias(self):
        modified = deepcopy(self.__neuron)
        modified.editConfig(bias=5-self.__neuron['bias'])
        self.assertEqual([q+5 for q in self.__neuron['q']], modified['q'])
        modified = deepcopy(self.__neuron)
        modified.editConfig(bias=-2-self.__neuron['bias'])
        self.assertEqual([q-2 for q in self.__neuron['q']], modified['q'])

    def test_08_identity_af(self):
        self.__neuron.setConfig({'f_act':'identity', 'q':[1 for _ in range(self.__n)], 'bias':0})
        self.__neuron.forward(self.__batch)
        self.assertEqual([-2.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 2.0], self.__neuron.output)

    def test_09_step_af(self):
        self.__neuron.setConfig({'f_act':'step', 'q':[1 for _ in range(self.__n)], 'bias':0})
        self.__neuron.forward(self.__batch)
        self.assertEqual([0, 0, 0, 0, 0, 1, 0, 1, 1], self.__neuron.output)

    def test_10_relu(self):
        self.__neuron.setConfig({'f_act':'relu', 'q':[1 for _ in range(self.__n)], 'bias':0})
        self.__neuron.forward(self.__batch)
        self.assertEqual([0, 0, 0, 0, 0, 1.0, 0, 1.0, 2.0], self.__neuron.output)

    def test_11_sigmoid_af(self):
        self.__neuron.setConfig({'f_act':'sigmoid', 'q':[1 for _ in range(self.__n)], 'bias':0})
        self.__neuron.forward(self.__batch)
        self.assertEqual([
                0.11920292202211755, 0.2689414213699951, 0.5, 0.2689414213699951,
                0.5, 0.7310585786300049, 0.5, 0.7310585786300049, 0.8807970779778823
            ], self.__neuron.output
        )

    def test_12_tanh_af(self):
        self.__neuron.setConfig({'f_act':'tanh', 'q':[1 for _ in range(self.__n)], 'bias':0})
        self.__neuron.forward(self.__batch)
        self.assertEqual([-0.964027580075817, -0.7615941559557649, 0.0, -0.7615941559557649,
                0.0, 0.7615941559557649, 0.0, 0.7615941559557649, 0.964027580075817
            ], self.__neuron.output
        )

    def test_13_q_and_n_err(self):
        with self.assertRaises(SystemExit):
            self.__neuron.editConfig(n_inputs=3, q=[1.1, 1.2])

    def test_14_af_err(self):
        with self.assertRaises(SystemExit):
            self.__neuron.editConfig(f_act='foobar')


if __name__ == '__main__':
    print('{}-I-{} Running unit tests on {}Neuron{} class...'.format(
        helpers.Format['GREEN'], helpers.Format['END'],
        helpers.Format['BOLD'], helpers.Format['END']
    ))
    print('----------------------------------------------------------------------')
    unittest.main(verbosity=2)
