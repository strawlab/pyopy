# coding=utf-8
from functools import partial

import pytest

from pyopy.hctsa.hctsa_utils import prepare_engine_for_hctsa
from pyopy.matlab_utils import Oct2PyEngine, PyMatBridgeEngine


@pytest.yield_fixture(scope='module', params=['oct2py', 'pymatbridge-oct', 'pymatbridge-mat'])
def eng(request):
    engines = {'oct2py': Oct2PyEngine,
               'pymatbridge-mat': partial(PyMatBridgeEngine, octave=False),
               'pymatbridge-oct': partial(PyMatBridgeEngine, octave=True)}
    with engines[request.param]() as eng:
        prepare_engine_for_hctsa(eng)
        yield eng


def test_outputs(eng):
    pass