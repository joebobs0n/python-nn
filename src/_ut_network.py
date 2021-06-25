#!/usr/bin/python3
r'''
* --- NETWORK CLASS UNIT TEST ---------------------------------------------------------------------
* -------------------------------------------------------------------------------------------------
* Unit testing to cover trivial and non-trivial features of Network class.
*
* Author: joebobs0n
* Last Edited: 25 Jun 2021
* -------------------------------------------------------------------------------------------------
'''

from network import Network
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
