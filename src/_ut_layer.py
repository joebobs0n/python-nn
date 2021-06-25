#!/usr/bin/python3
r'''
* --- LAYER CLASS UNIT TEST -----------------------------------------------------------------------
* -------------------------------------------------------------------------------------------------
* Unit testing to cover trivial and non-trivial features of Layer class.
*
* Author: joebobs0n
* Last Edited: 25 Jun 2021
* -------------------------------------------------------------------------------------------------
'''

from layer import Layer
from copy import deepcopy
import helpers
import unittest


class TestLayer(unittest.TestCase):
    def setUp(self):
        pass


if __name__ == '__main__':
    print('{}-I-{} Running unit tests on {}Layer{} class...'.format(
        helpers.Format['GREEN'], helpers.Format['END'],
        helpers.Format['BOLD'], helpers.Format['END']
    ))
    print('----------------------------------------------------------------------')
    unittest.main(verbosity=2)
