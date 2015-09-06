# coding=utf-8
from __future__ import print_function

"""Tools to extracts features using the HCTSA python bindings.
Assumes the bindings have already been generated.
"""
import inspect
import time

import numpy as np

from pyopy.hctsa.hctsa_bindings import CO_AddNoise, DN_Cumulants, WL_fBM, WL_scal2frq
from pyopy.hctsa.hctsa_install import hctsa_prepare_engine
from pyopy.hctsa.hctsa_transformers import hctsa_prepare_input
from pyopy.base import py2matstr, PyopyEngines


# ----- Octave code generation

def as_partial_call(hctsa_feat):
    """
    Returns a matlab anon function declaration with a partial application corresponding to the hctsa feature.

    Examples
    --------
    >>> as_partial_call(CO_AddNoise())
    "(@(x)CO_AddNoise(x, 1.0, 'quantiles', 20.0))"
    >>> as_partial_call(CO_AddNoise(tau='tau', meth='blah', nbins=-1))
    "(@(x)CO_AddNoise(x, 'tau', 'blah', -1))"
    """
    func_name = hctsa_feat.__class__.__name__
    parameters_sorted_as_in_matlab_call = inspect.getargspec(hctsa_feat.__init__)[0][1:]
    parameters = []
    for param in parameters_sorted_as_in_matlab_call:
        val = hctsa_feat.__getattribute__(param)
        if val is not None:
            parameters.append(val)
        else:
            break
    if 0 == len(parameters):
        return '(@(x)%s(x))' % func_name
    return '(@(x)%s(x, %s))' % (func_name, ', '.join(map(py2matstr, parameters)))


# ----- Output postprocessing

def flatten_hctsa_result(result, name=''):
    """Outputs of an HCTSA feature extraction into a 1D numpy array."""
    if isinstance(result, dict):
        columns = sorted(result.keys())
        return ['out=%s#%s' % (column, name) for column in columns], np.array([result[k] for k in columns])
    return name, np.ndarray(result)


# ----- High-level usage of the library

def hctsa_partials_poc(eng_thunk=PyopyEngines.matlab,
                       data=None,
                       # A few hctsa features chosen for no reason
                       features=(CO_AddNoise(),
                                 DN_Cumulants(cumWhatMay='skew1'),
                                 DN_Cumulants(cumWhatMay='skew2'),
                                 DN_Cumulants(cumWhatMay='kurt1'),
                                 DN_Cumulants(cumWhatMay='kurt2'),
                                 WL_fBM(),
                                 WL_scal2frq(wname='db3'),
                                 WL_scal2frq(wname='sym2', amax='max', delta=10))):

    with eng_thunk() as eng:

        # Prepare for HCTSA
        hctsa_prepare_engine(eng)

        # If data is None, test on fake data
        if data is None:
            ne = 100
            rng = np.random.RandomState(2147483647)
            data = [rng.randn(rng.randint(100, 10000), 1) for _ in range(10)]
            print(','.join(map(str, map(len, data))))
        else:
            ne = len(data)

        # Make sure each series is suitable for analysis with HCTSA
        data = map(hctsa_prepare_input, data)

        # To matlab-land
        eng.put(u'data', [data])
        print('type of data in matlab-land:', eng.engine_class(u'data'))

        # Compute features
        start = time.time()
        results = {}
        for feature in features:
            partial = as_partial_call(feature)
            response, result = eng.eval(u'cellfun(%s, data, \'UniformOutput\', 0)' % partial,
                                        outs2py=True)
            if not response.success:
                raise Exception(response.stdout)
            results[feature.who().id()] = result
        print('Taken %.2f seconds' % (time.time() - start))
        # results to matrix
        matrix = []
        fnames = []
        for fname, fvals in results.items():
            print(fname)
            # extract from list returned by run_command
            fvals = fvals[0]
            print(type(fvals), len(fvals))
            if isinstance(fvals, np.ndarray):
                fnames.append(fname)
                fvals = fvals[0]
                assert len(fvals) == ne
                matrix.append(fvals)
            else:
                out_names = fvals[0].dtype.names
                print('num-outputs = %d' % len(out_names))
                for out_name in out_names:
                    fnames.append('%s#out=%s' % (fname, out_name))
                    v = np.array([fval[out_name][0][0][0] for fval in fvals])  # LAME
                    assert len(fvals) == ne
                    matrix.append(v)
        X = np.array(matrix).T  # Make it recarray...
        # ...or better, pandas
        import pandas as pd
        df = pd.DataFrame(data=X, columns=fnames)
        # print df.describe()
        return df


######################################
# TODOs
######################################
#
# TODO: batch processing:
#       allow to generate many call lines to group the operations,
#       run them in data already in matlab-land, so we get rid of the excessive call overhead
#       probably use one of (rowfun, colfun, cellfun) in matlab land, put in matlab_utils
#       Interesting :
#          https://github.com/adambard/functools-for-matlab
#          (lambdas are slow in matlab)
#       cellfun is slow in matlab, usually much slower than pathetically slow loops
#          http://www.mathworks.com/matlabcentral/newsreader/view_thread/253815
#          http://stackoverflow.com/questions/18284027/cellfun-versus-simple-matlab-loop-performance
#
# TODO: estimate speed / complexity (hard because it can also depend on outputs)
#
# TODO: create tests with outputs from matlab
#
######################################
# try...catch works in both octave and matlab
# memmaps are only supported in matlab
######################################
