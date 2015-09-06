# coding=utf-8
import pytest
from pyopy.base import PyopyEngines
from pyopy.hctsa.hctsa_install import hctsa_prepare_engine


@pytest.yield_fixture(scope='module', params=['matlab', 'octave'])
def eng(request):
    engines = {'octave': PyopyEngines.octave,
               'matlab': PyopyEngines.matlab}
    with engines[request.param]() as eng:
        hctsa_prepare_engine(eng)
        yield eng


def test_outputs(eng):
    pass
