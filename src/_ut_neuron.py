#!/usr/bin/python3
r'''
* --- NEURON CLASS UNIT TEST ----------------------------------------------------------------------
* -------------------------------------------------------------------------------------------------
* Unit testing to cover trivial and non-trivial features of Neuron class.
*
* Author: joebobs0n
* Last Edited: 25 Jun 2021
* -------------------------------------------------------------------------------------------------
'''

from neuron import Neuron
from copy import deepcopy
import helpers
import unittest


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

    def test_01_defaults(self):
        special_cases = ['n_inputs', 'act_f', 'q']
        for key, val in self.__defaults.items():
            if key not in special_cases:
                self.assertEqual(val, self.__neuron[key])
            elif key == 'act_f':
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
            'act_f': 'identity',
            'q': [1, 2, 3, 4, 5],
            'q_scalar': -1,
            'bias': 3.14,
            'output': []
        }
        self.__neuron.setConfig(config)
        special_cases = ['act_f', 'q']
        for tempConfig1, tempConfig2 in zip(config.items(), self.__neuron.config.items()):
            key, val1 = tempConfig1
            _, val2 = tempConfig2
            if key not in special_cases:
                self.assertEqual(val1, val2)
            elif key == 'act_f':
                f1 = val1.lower() if type(val1) == str else val1.__name__[2:]
                f2 = val2.lower() if type(val2) == str else val2.__name__[2:]
                self.assertEqual(f1, f2)
            elif key == 'q':
                self.assertEqual(len(val1), len(val2))

    def test_05_sparce_config(self):
        configs = [
            {'n_inputs': 5},
            {'act_f': 'tanh'},
            {'q': [0.3, -1.2, 1.3, 8.0, -23]},
            {'q_scalar': -1},
            {'bias': 1}
        ]
        for c in configs:
            key, val = list(c.items())[0]
            modified = deepcopy(self.__neuron)
            modified.setConfig(c)
            if key == 'n_inputs':
                self.assertEqual(val, modified['n_inputs'])
                self.assertEqual(val, len(modified['q']))
            elif key == 'act_f':
                self.assertEqual(val.lower(), modified['act_f'].__name__[2:])
            elif key == 'q':
                self.assertEqual(len(val), modified['n_inputs'])
                self.assertEqual(val, modified['q'])
            elif key == 'q_scalar':
                self.assertEqual(val, modified['q_scalar'])
            elif key == 'bias':
                self.assertEqual(val, modified['bias'])

    def test_06_q_scalar(self):
        self.__neuron.editConfig(act_f='identity')
        modified = deepcopy(self.__neuron)
        modified.editConfig(q_scalar=self.__neuron['q_scalar']/2)
        self.assertEqual(
            [o/2 for o in self.__neuron.forward(self.__batch)],
            modified.forward(self.__batch)
        )
        modified = deepcopy(self.__neuron)
        modified.editConfig(q_scalar=self.__neuron['q_scalar']*-1)
        self.assertEqual(
            [o*-1 for o in self.__neuron.forward(self.__batch)],
            modified.forward(self.__batch)
        )

    def test_07_bias(self):
        new_bias = 5
        self.__neuron.editConfig(bias=new_bias)
        self.assertEqual(new_bias, self.__neuron['bias'])

    def test_08_identity_af(self):
        self.__neuron.setConfig({'act_f':'identity', 'q':[1 for _ in range(self.__n)], 'bias':0})
        self.__neuron.forward(self.__batch)
        self.assertEqual([-2.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 2.0], self.__neuron.output)

    def test_09_step_af(self):
        self.__neuron.setConfig({'act_f':'step', 'q':[1 for _ in range(self.__n)], 'bias':0})
        self.__neuron.forward(self.__batch)
        self.assertEqual([0, 0, 0, 0, 0, 1, 0, 1, 1], self.__neuron.output)

    def test_10_relu(self):
        self.__neuron.setConfig({'act_f':'relu', 'q':[1 for _ in range(self.__n)], 'bias':0})
        self.__neuron.forward(self.__batch)
        self.assertEqual([0, 0, 0, 0, 0, 1.0, 0, 1.0, 2.0], self.__neuron.output)

    def test_11_sigmoid_af(self):
        self.__neuron.setConfig({'act_f':'sigmoid', 'q':[1 for _ in range(self.__n)], 'bias':0})
        self.__neuron.forward(self.__batch)
        self.assertEqual([
                0.11920292202211755, 0.2689414213699951, 0.5, 0.2689414213699951,
                0.5, 0.7310585786300049, 0.5, 0.7310585786300049, 0.8807970779778823
            ], self.__neuron.output
        )

    def test_12_tanh_af(self):
        self.__neuron.setConfig({'act_f':'tanh', 'q':[1 for _ in range(self.__n)], 'bias':0})
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
            self.__neuron.editConfig(act_f='foobar')


if __name__ == '__main__':
    print('{}-I-{} Running unit tests on {}Neuron{} class...'.format(
        helpers.Format['GREEN'], helpers.Format['END'],
        helpers.Format['BOLD'], helpers.Format['END']
    ))
    print('----------------------------------------------------------------------')
    unittest.main(verbosity=2)
