#!/usr/bin/python3
r'''

'''

try:
    import src.helpers as helpers
    from src.neuron import Neuron
except ModuleNotFoundError:
    import helpers
    import unittest
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


    #? --- PROPERTIES -----------------------------------------------------------------------------
    #? --------------------------------------------------------------------------------------------


#! --- UNIT TEST ----------------------------------------------------------------------------------
#! ------------------------------------------------------------------------------------------------
#!
#! ------------------------------------------------------------------------------------------------

class TestLayer(unittest.TestCase):
    def setUp(self):
        pass

if __name__ == '__main__':
    print('{}-I-{} Running unit tests on {}Layer{} class...'.format(
        helpers.Format['GREEN'], helpers.Format['END'],
        helpers.Format['BOLD'], helpers.Format['END']
    ))
    print('----------------------------------------------------------------------')
    unittest.main()
