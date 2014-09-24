# coding=utf-8
import time
import inspect

import numpy as np
import oct2py

from pyopy.hctsa.bindings.hctsa_bindings import HCTSA_SY_SpreadRandomLocal, CO_AddNoise, DN_Cumulants, WL_fBM, \
    WL_scal2frq
from pyopy.hctsa.hctsa_utils import prepare_engine_for_hctsa
from pyopy.matlab_utils import Oct2PyEngine, py2matstr, PyMatBridgeEngine


def as_partial_call(hctsa_feat, engine=None, partials_cache={}):
    if hctsa_feat.configuration().id() not in partials_cache:
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
            command = '(@(x)%s(x))' % func_name
        else:
            command = '(@(x)%s(x, %s))' % (func_name, ', '.join(map(py2matstr, parameters)))

        if engine is None:
            return command
        # Now, if we are provided an engine, we will put the function call in octave-land and cache it in memory
        raise NotImplementedError('Still need to implement getting a partial in octave-land')
        # engine.run_text()
    return partials_cache.get(hctsa_feat.configuration().id())


def hctsa_partials_poc(eng='matlab',
                       data=None,
                       # A few hctsa features chosen for no reason
                       features=(CO_AddNoise(),
                                 DN_Cumulants(whatcum='skew1'),
                                 DN_Cumulants(whatcum='skew2'),
                                 DN_Cumulants(whatcum='kurt1'),
                                 DN_Cumulants(whatcum='kurt2'),
                                 WL_fBM(),
                                 WL_scal2frq(wname='db3'),
                                 WL_scal2frq(wname='sym2', amax='max', delta=10))):

    with (Oct2PyEngine() if eng == 'octave' else PyMatBridgeEngine()) as eng:

        # Prepare for HCTSA
        prepare_engine_for_hctsa(eng)

        # Some fake data here
        if data is None:
            ne = 100
            rng = np.random.RandomState(2147483647)
            data = [rng.randn(rng.randint(100, 10000)) for _ in xrange(10)]
            print ','.join(map(str, map(len, data)))
        else:
            ne = len(data)

        # To matlab-land
        eng.put(u'data', [data])
        print 'type of data in matlab-land:', eng.matlab_class(u'data')

        # Compute features
        start = time.time()
        results = {}
        for feature in features:
            partial = as_partial_call(feature)
            response, result = eng.run_command(u'cellfun(%s, data, \'UniformOutput\', 0)' % partial,
                                               outs2py=True)
            if not response.success:
                raise Exception(response.stdout)
            results[feature.configuration().id()] = result
        print 'Taken %.2f seconds' % (time.time() - start)
        # results to matrix +
        matrix = []
        fnames = []
        for fname, fvals in results.iteritems():
            print fname
            # extract from list returned by run_command
            fvals = fvals[0]
            print type(fvals), len(fvals)
            if isinstance(fvals, np.ndarray):
                fnames.append(fname)
                fvals = fvals[0]
                assert len(fvals) == ne
                matrix.append(fvals)
            else:
                out_names = fvals[0].dtype.names
                print 'num-outputs = %d' % len(out_names)
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


def arrays2cells_and_partial(eng='octave'):

    eng = Oct2PyEngine() if eng == 'octave' else PyMatBridgeEngine()

    # Prepare for HCTSA
    prepare_engine_for_hctsa(eng)

    # Generate data
    arrays = [np.random.randn(size) for size in (100, 500, 100, 35, 200, 130, 230)]
    # To octave land
    _ = eng.put('x', [arrays])  # N.B. needs to be list to make a cell

    # cellfun passing lambda (can be useful for partial application and to aggregate operators)
    start = time.time()
    eng.run_command('f1=@(x) SY_SpreadRandomLocal(x, \'ac5\')')
    response, result = eng.run_command('ans=cellfun(f1, x, \'UniformOutput\', 0)', outs2py=True)
    print response.success, response.stdout
    print result
    print 'cellfun with partial took %.2f seconds' % (time.time() - start)
    # cellfun with partial took 5.42 seconds (octave)
    # cellfun with partial took 1.12 seconds (matlab) - bye oct2py call overhead + octave slower (use faster builds)

    # python-land loop
    taken = 0
    result = []
    for array in arrays:
        array = eng.put('blah', array)
        start = time.time()
        result.append(HCTSA_SY_SpreadRandomLocal(eng, array, l='ac5'))
        taken += time.time() - start
    print result
    print 'python-land loop took %.2f seconds' % taken
    # python-land loop took 6.18 seconds (octave)
    # python-land loop took 1.27 seconds (matlab) - bye oct2py call overhead + octave slower (use faster builds)


if __name__ == '__main__':

    # arrays2cells_and_partial(eng='matlab')
    # arrays2cells_and_partial(eng='octave')
    hctsa_partials_poc()
