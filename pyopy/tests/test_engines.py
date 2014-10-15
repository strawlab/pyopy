# coding=utf-8
from functools import partial
from pyopy.hctsa.hctsa_utils import hctsa_sine
from pyopy.matlab_utils import Oct2PyEngine, PyMatBridgeEngine, MatlabSequence
import numpy as np
import pytest


@pytest.yield_fixture(scope='module', params=['oct2py', 'pymatbridge-oct', 'pymatbridge-mat'])
def eng(request):
    engines = {'oct2py': Oct2PyEngine,
               'pymatbridge-mat': partial(PyMatBridgeEngine, octave=False),
               'pymatbridge-oct': partial(PyMatBridgeEngine, octave=True)}
    with engines[request.param]() as eng:
        yield eng


def test_roundtrip_scalar(eng):

    # Automatic cast to double
    val = eng.put('a', 12)
    assert val.name == 'a'
    assert val.get() == 12
    assert eng.get('a') == 12
    assert val.matlab_class() == u'double'
    assert val.exists()
    val.clear()
    assert not val.exists()

    # Same, without automatic cast
    eng.int2float = False  # This is nasty, part 1
    val = eng.put('a', 12)
    eng.int2float = True  # This is nasty, part 2
    assert val.name == 'a'
    assert val.get() == 12
    assert eng.get('a') == 12
    assert val.matlab_class() == u'int64'
    assert val.exists()
    val.clear()
    assert not val.exists()

    val = eng.put('b', 14.)
    assert val.name == 'b'
    assert val.get() == 14.
    assert eng.get('b') == 14.
    assert val.matlab_class() == u'double'
    assert val.exists()
    val.clear()
    assert not val.exists()

    val = eng.put('c', 'lala')
    assert val.name == 'c'
    assert val.get() == 'lala'
    assert eng.get('c') == 'lala'
    assert val.matlab_class() == u'char'
    assert val.exists()
    val.clear()
    assert not val.exists()


def test_put_get_clear_many(eng):

    x1, x2 = eng.put(('x1', 'x2'), (100, MatlabSequence('1:20')))

    assert x1.name == 'x1'
    assert x1.get() == 100
    assert eng.get('x1') == 100
    assert x1.matlab_class() == u'double'  # magic autoconversion
    assert x1.exists()

    assert x2.name == 'x2'
    assert x2.get().shape == (1, 20)
    assert np.all(eng.get('x2') == MatlabSequence('1:20').as_array())
    assert x2.matlab_class() == u'double'

    x1, x2 = eng.get(('x1', 'x2'))
    assert x1 == 100
    assert np.all(x2 == MatlabSequence('1:20').as_array())

    eng.clear(('x1', 'x2'))
    assert not eng.exists('x1')
    assert not eng.exists('x2')


def test_roundtrip_sequence(eng):

    val = eng.put('d', MatlabSequence('1:80'))
    assert val.name == 'd'
    d = val.get()
    assert d.shape == (1, 80)
    assert np.all(np.arange(1, 81) == d)
    assert val.exists()
    assert val.matlab_class() == u'double'
    val.clear()
    assert not val.exists()


def test_roundtrip_array(eng):

    x_py = hctsa_sine()
    x_mat = eng.put('x', x_py).get()

    assert np.isclose(x_py.mean(), x_mat.mean())
    assert np.isclose(x_py.std(), x_mat.std())
    assert np.allclose(x_py, x_mat)

    # Now let's z-score it
    x_py = (x_py - x_py.mean()) / (x_py.std())
    x_mat = eng.put('x', x_py).get()

    assert np.isclose(x_py.mean(), x_mat.mean())
    assert np.isclose(x_py.std(), x_mat.std())
    assert np.allclose(x_py, x_mat)

    # But actually matlab's computed std differs...
    # See http://stackoverflow.com/questions/7482205/precision-why-do-matlab-and-python-numpy-give-so-different-outputs


def test_run_outs2py(eng):

    response, result = eng.run_command('a=[1,2]', outs2py=True)
    assert response.success
    assert len(result) == 1
    assert result[0].shape == (1, 2)
    assert np.allclose(result, np.array([1, 2]))
    assert eng.exists('a')
    eng.clear('a')
    assert not eng.exists('a')

    response, result = eng.run_command('[x, y]=meshgrid(5, 4)', outs2py=True)
    assert response.success
    assert len(result) == 2
    x, y = result
    assert x == 5
    assert y == 4
    assert eng.exists('x')
    eng.clear('x')
    assert not eng.exists('x')
    assert eng.exists('y')
    eng.clear('y')
    assert not eng.exists('y')


def test_run_function(eng):

    # matlab sequences as parameters
    result = eng.run_function(1, 'sum', MatlabSequence('1:80'))
    assert result == MatlabSequence('1:80').as_array().sum()


####################
#
# --- complete roundtrip tests
#     (go and copy from oct2py, pymatbridge and the like, they are comprehensive there...)
#
####################
