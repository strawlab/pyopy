# coding=utf-8
from pyopy.matlab_utils import Oct2PyEngine, PyMatBridgeEngine


def test_oct2py_engine():
    engine = Oct2PyEngine()
    val = engine.put('a', 12)
    assert val.get() == 12


def test_pymatbridge_engine():
    with PyMatBridgeEngine() as eng:
        eng.run_command('a=3')
        print eng.matlab_class('a')
        assert eng.get('a', strategy='by_file') == 3
        assert eng.get('a', strategy=None) == 3
        eng.run_command('b=randn(3,3)')
        assert eng.get('b', strategy='by_file').shape == (3, 3)

        # wow this returns a list of lists?!?! TODO: recheck pymatlab API to see if we can coerce to ndarray
        # assert eng.get('b', strategy=None).shape == (3, 3)

        a, b = eng.get(('a', 'b'))
        assert a == 3
        assert b.shape == (3, 3)

        print eng.mwho()

        var = eng.put('c', 'lala')
        assert var.varname == 'c'
        assert var.get() == 'lala'

        a, b = eng.run_function(2, 'meshgrid', 5, 4)
        assert a == 5
        assert b == 4

        import numpy as np
        eng.put(('x', 'y'), (np.ones(10000), np.ones(10000) * 2))
        _, response = eng.run_command('ans=x.*y;', outs2py=True)
        print eng.get('x').shape
        assert eng.get('x').shape == (1, 10000)
        assert response[0].shape == (1, 10000)

        import time
        start = time.time()
        eng.run_command('ans=x.*y;', outs2py=False)
        eng.run_command('ones(2)')
        print 'Taken %.2f seconds' % (time.time() - start)

        # val = eng.put('a', 12)
        # assert val.get() == 12

test_pymatbridge_engine()
test_oct2py_engine()


####################
#
# from time import time
# with Oct2PyEngine() as eng:
#
#     # roundtrip tests (copy from oct2py...)
#     response, result = eng.run_command('a=[1,2]', outs2py=True)
#     assert np.allclose(result, np.array([1, 2]))
#
#     # who call
#     start = time()
#     eng.run_command('meshgrid(4,5)', outs2py=False)
#     print 'simplemost call took %.2f seconds' % (time() - start)
#
#     # function call - slow as hell because of the overhead in oct2py + overhead in Oct2PyEngine
#     start = time()
#     assert np.allclose(eng.run_function(2, 'meshgrid', 4, 5), [4, 5])
#     print 'function overhead is like call took %.2f seconds' % (time() - start)
#
#     # which variables are at the moment there?
#     print 'variables in octave workspace:', eng.who()
#
#     exit(33)
#
####################


#
# TODO: an abstract tester class, or use parameterized fixtures to test all engines with the same test suite
# see:  http://pytest.org/latest/parametrize.html
#