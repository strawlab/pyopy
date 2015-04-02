# coding=utf-8
from functools import partial

import numpy as np
import pytest
from pyopy.backend_mathworks import MathworksEngine, MathworksTransplanter
from pyopy.backend_matlab_wrapper import MatlabWrapperEngine, MatlabWrapperTransplanter
from pyopy.backend_oct2py import Oct2PyEngine
from pyopy.backend_pymatbridge import PyMatBridgeEngine, PyMatBridgeTransplanter

from pyopy.base import MatlabSequence
from pyopy.hctsa.hctsa_data import hctsa_sine


@pytest.yield_fixture(scope='module', params=['oct2py',
                                              # pymatbridge is quite unreliable as per 2015/02/11
                                              # 'pymatbridge-oct',
                                              # 'pymatbridge-mat',
                                              # 'pymatbridge-oct-pmb',
                                              # 'pymatbridge-mat-pmb',
                                              # matlabwrapper dispatch is too slow as per 2015/02/11
                                              # 'matlabwrapper',
                                              # 'matlabwrapper-mwr',
                                              # Mathworks dispatch rocks, data transfer with oct2py is perfect combo
                                              'mathworks',
                                              # 'mathworks-mathworks',
                                              ])
def eng(request):
    engines = {'oct2py': Oct2PyEngine,
               'pymatbridge-mat': partial(PyMatBridgeEngine, octave=False),
               'pymatbridge-oct': partial(PyMatBridgeEngine, octave=True),
               'oct2py-pmb': partial(Oct2PyEngine, transplanter=PyMatBridgeTransplanter()),
               'pymatbridge-mat-pmb': partial(PyMatBridgeEngine, octave=False, transplanter=PyMatBridgeTransplanter()),
               'pymatbridge-oct-pmb': partial(PyMatBridgeEngine, octave=True, transplanter=PyMatBridgeTransplanter()),
               'matlabwrapper': MatlabWrapperEngine,
               'matlabwrapper-mwr': partial(MatlabWrapperEngine, transplanter=MatlabWrapperTransplanter()),
               'mathworks': MathworksEngine,
               'mathworks-mathworks': partial(MatlabWrapperEngine, transplanter=MathworksTransplanter())}
    with engines[request.param]() as eng:
        yield eng


def test_roundtrip_scalar(eng):

    # Automatic cast to double
    val = eng.put('a', 12)
    assert val.name == 'a'
    assert val.get() == 12
    assert eng.get('a') == 12
    assert val.engine_class() == u'double'
    assert val.exists()
    val.clear()
    assert not val.exists()

    # Same, without automatic cast
    val = eng.put('a', 12, int2float=False)
    assert val.name == 'a'
    assert val.get() == 12
    assert eng.get('a') == 12
    assert val.engine_class() == u'int64'
    assert val.exists()
    val.clear()
    assert not val.exists()

    val = eng.put('b', 14.)
    assert val.name == 'b'
    assert val.get() == 14.
    assert eng.get('b') == 14.
    assert val.engine_class() == u'double'
    assert val.exists()
    val.clear()
    assert not val.exists()

    val = eng.put('c', 'lala')
    assert val.name == 'c'
    assert val.get() == 'lala'
    assert eng.get('c') == 'lala'
    assert val.engine_class() == u'char'
    assert val.exists()
    val.clear()
    assert not val.exists()


def test_put_get_clear_many(eng):

    x1, x2 = eng.put(('x1', 'x2'), (100, MatlabSequence('1:20')))

    assert x1.name == 'x1'
    assert x1.get() == 100
    assert eng.get('x1') == 100
    assert x1.engine_class() == u'double'  # magic autoconversion
    assert x1.exists()

    assert x2.name == 'x2'
    assert x2.get().shape == (1, 20)
    assert np.all(eng.get('x2') == MatlabSequence('1:20').as_array())
    assert x2.engine_class() == u'double'

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
    assert val.engine_class() == u'double'
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


def test_roundtrip_cell(eng):

    # Mixed cells, as tuple
    cell_tuple_mixed = ('ac', 5)
    eng.put('cell_tuple_mixed', [cell_tuple_mixed])
    assert eng.get('cell_tuple_mixed') == cell_tuple_mixed
    assert eng.engine_class('cell_tuple_mixed') == 'cell'

    # Recursive cells (FIXME: hackish implementation)
    cellr = ('covSum', ('covSEiso', 'covNoise'))
    # this is quite bad, and won't play nice with e.g. matlabwrapper that does the right thing...
    wrong_roundtripped_expected = ['covSum', np.array(['covSEiso', 'covNoise'])]
    eng.put('cellr', [cellr])
    roundtripped = eng.get('cellr')
    assert roundtripped[0] == wrong_roundtripped_expected[0]
    assert (roundtripped[1] == wrong_roundtripped_expected[1]).all()
    assert eng.engine_class('cellr') == 'cell'

    # Homogeneous-type cell, as list
    cell_list_nonummeric = ['ami1', 'fmmi', 'o3', 'tc3']
    eng.put('cell_list_nonummeric', [cell_list_nonummeric])
    assert eng.get('cell_list_nonummeric') == cell_list_nonummeric
    assert eng.engine_class('cell_list_nonummeric') == 'cell'

    # TODO: using a string to avoid pure-numeric cells to be passed as arrays is dirty
    #       it would be better (even if more expensive) to have a Cell class in python-land
    cell2 = [1, 2, 'c']
    eng.put('cell2', [cell2])
    assert eng.get('cell2') == cell2
    assert eng.engine_class('cell_list_nonummeric') == 'cell'

    # TODO: test that a numeric cell gets converted to (..., '_celltrick_')


def test_run_outs2py(eng):

    response, result = eng.eval('a=[1,2]', outs2py=True)
    assert response.success
    assert len(result) == 1
    assert result[0].shape == (1, 2)
    assert np.allclose(result, np.array([1, 2]))
    assert eng.exists('a')
    eng.clear('a')
    assert not eng.exists('a')

    response, result = eng.eval('[x, y]=meshgrid(5, 4)', outs2py=True)
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
