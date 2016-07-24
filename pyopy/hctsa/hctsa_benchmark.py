# coding=utf-8
"""Benchmarks and checks the HCTSA python bindings."""
from itertools import product, izip
import os.path as op
import random
import time
from glob import glob
from datetime import datetime
from socket import gethostname
from lockfile import LockFile

import pandas as pd
import numpy as np

from pyopy.base import PyopyEngines, EngineException
from pyopy.config import PYOPY_EXTERNAL_TOOLBOXES_DIR
from pyopy.hctsa.hctsa_bindings import HCTSAOperations
from pyopy.hctsa.hctsa_catalog import HCTSACatalog
from pyopy.hctsa.hctsa_data import hctsa_sine, hctsa_noise, hctsa_noisysinusoid
from pyopy.hctsa.hctsa_install import hctsa_prepare_engine
from pyopy.hctsa.hctsa_transformers import hctsa_prepare_input
from pyopy.misc import ensure_dir


# Where benchmark and check results will live
HCTSA_BENCHMARKS_DIR = op.join(PYOPY_EXTERNAL_TOOLBOXES_DIR, 'hctsa_benchmarks')

# Operations that can potentially hang the system - which BTW should be properlly fixed
HCTSA_FORBIDDEN_OPERATIONS = {
    'Oct2PyEngine': (('HCTSA_MF_ARMA_orders', 'Enters Oct2Py interact mode'),
                     ('HCTSA_SY_DriftingMean', 'Has a bug (l is not defined) and enters Oct2Py interact mode'),
                     ('HCTSA_TSTL_predict', 'Takes too long?',)),
}

# Time series
TS_FACTORIES = {
    'sine': hctsa_sine,
    'noise': hctsa_noise,
    'noisysinusoid': hctsa_noisysinusoid
}


def check_benchmark_bindings(x,
                             xname,
                             engine='matlab',
                             n_jobs=None,
                             transferinfo='ramdisk',
                             extra=None,
                             operations=None,
                             forbidden=None,
                             dest_file=None,
                             random_order=True):

    # Setup the engine
    engine = PyopyEngines.engine_or_matlab_or_octave(engine)
    hctsa_prepare_engine(engine)

    # Setup the input for hctsa
    size = len(x)
    y = engine.put('y', hctsa_prepare_input(x, z_scored=True))
    x = engine.put('x', hctsa_prepare_input(x, z_scored=False))

    # Operations
    if operations is None:
        operations = HCTSAOperations.all()

    # Some operations that make the entire experiment fail
    if forbidden is None:
        forbidden = dict(HCTSA_FORBIDDEN_OPERATIONS.get(engine.__class__.__name__, ()))

    # The host
    hostname = gethostname()

    # Number of threads used in the engine
    max_comp_threads = engine.max_comp_threads()

    # tidy results
    results = {
        'host': [],
        'engine': [],
        'transplanter': [],
        'transferinfo': [],
        'date': [],
        'extra': [],
        'n_jobs': [],
        'n_threads_matlab': [],
        'xname': [],
        'size': [],
        'taken_s': [],
        'operator': [],
        'operation': [],
        'output': [],
        'value': [],
        'error': []
    }

    #
    # We want to check what is deterministic and what is not
    # Newer versions of matlab are "deterministic" (always init the rng equally)
    # So we will always get the same result if we use them in the same order...
    # Let's just randomise the order of operations, useing clock-based seeded rng...
    #
    if random_order:
        random.Random().shuffle(operations)

    for opname, operation in operations:
        print opname
        start = time.time()
        try:
            if opname in forbidden:
                raise EngineException(None, 'Forbidden operation')
            result = operation.transform(y if HCTSACatalog.catalog().must_standardize(opname) else x, engine)
            taken = time.time() - start
            if not isinstance(result, dict):
                result = {None: result}
            for outname, fval in sorted(result.items()):
                results['host'].append(hostname)
                results['engine'].append(engine.__class__.__name__)  # use whatami
                results['transplanter'].append(engine.transplanter.__class__.__name__)
                results['transferinfo'].append(transferinfo)
                results['date'].append(datetime.now())
                results['extra'].append(extra)
                results['n_jobs'].append(n_jobs)
                results['n_threads_matlab'].append(max_comp_threads)
                results['xname'].append(xname)
                results['size'].append(size)
                results['taken_s'].append(taken)
                results['operator'].append(operation.__class__.__name__)
                results['operation'].append(opname)
                results['output'].append(outname)
                results['value'].append(fval)
                results['error'].append(None)
        except EngineException as engex:
            taken = time.time() - start
            results['host'].append(hostname)
            results['engine'].append(engine.__class__.__name__)  # use whatami
            results['transplanter'].append(engine.transplanter.__class__.__name__)
            results['transferinfo'].append(transferinfo)
            results['date'].append(datetime.now())
            results['extra'].append(extra)
            results['n_jobs'].append(n_jobs)
            results['n_threads_matlab'].append(max_comp_threads)
            results['xname'].append(xname)
            results['size'].append(size)
            results['taken_s'].append(taken)
            results['operator'].append(operation.__class__.__name__)
            results['operation'].append(opname)
            results['output'].append(np.nan)
            results['value'].append(None)
            results['error'].append(str(engex))

    # Save the dataframe
    df = pd.DataFrame(data=results)
    if dest_file is None:
        dest_file = op.join(HCTSA_BENCHMARKS_DIR, 'hctsa_checks_%s.pickle' % hostname)
        ensure_dir(op.dirname(dest_file))
    with LockFile(dest_file):  # lame inefficient incrementality
        if op.isfile(dest_file):
            df = pd.concat((pd.read_pickle(dest_file), df))
        df.to_pickle(dest_file)


def analyse():

    # Load all the results
    df = pd.concat(map(pd.read_pickle, glob(op.join(HCTSA_BENCHMARKS_DIR, '*.pickle'))))

    # Reorder columns
    columns = [u'host',
               u'date',
               u'engine',
               u'transplanter',
               u'transferinfo',
               u'n_jobs',
               u'n_threads_matlab',
               u'extra',
               u'xname',
               u'operation',
               u'operator',
               u'output',
               u'value',
               u'size',
               u'taken_s',
               u'error']
    df = df[columns]

    # Make some stuff categorical
    categoricals = ('host', 'engine', 'transplanter', 'transferinfo', 'extra', 'xname', 'operator', 'operation')
    for categorical in categoricals:
        df[categorical] = df[categorical].astype('category')  # there must be something in pandas to do this at once

    # One value was an empty list on one run, tisean routine, check (maybe concurrency?)
    def is_float(val):
        try:
            float(val)
            return True
        except:
            return False
    float_values = df['value'].apply(is_float)
    print '%d values were non-floats' % (~float_values).sum()
    print df[~float_values]['operation']
    df = df[float_values]
    df = df.convert_objects(convert_numeric=True)  # After removing these, value can be again converted to float

    nodup = df.dropna(subset=['error'], axis=0).drop_duplicates(['operator', 'error']).sort('operation')
    nodup.to_html(op.expanduser('~/dupes.html'))

    print '\n'.join(nodup['operator'].unique())
    operations_failing = map(lambda o: 'HCTSAOperations.%s[2],' % o, sorted(nodup['operation'].unique()))
    print '\n'.join(operations_failing)
    for opname, error in izip(nodup['operation'], nodup['error']):
        print '-' * 80
        print opname
        print error

    # Round to the 6th decimal
    df['value'] = np.around(df['value'], decimals=6)

    # Impact of running from pycharm

    # Infinities, but not explicit errors (N.B. pandas isnull does not take into account infinities)
    infinite = ~np.isfinite(df['value']) & ~np.isnan(df['value'])
    # Errors (but not infinities)
    nans = np.isnan(df['value'])
    # Failed
    failed = infinite | nans

    # Features that are stochastic
    catalog = HCTSACatalog.catalog()

    def stochastic_failing(df, verbose=False, tooverbose=False):
        for (xname, operation, output), oodf in df.groupby(['xname', 'operation', 'output']):
            operator = oodf['operator'].iloc[0]
            tagged_as_stochastic = catalog.operation(operation).has_tag('stochastic')
            failing = 'OK' if (~np.isfinite(oodf['value'])).sum() == 0 else 'FAILING'
            if oodf['value'].nunique() == 0:
                print xname, operator, operation, output, failing, failing, failing, tagged_as_stochastic
            elif oodf['value'].nunique() == 1:
                if failing != 'OK' or verbose:
                    print xname, operator, operation, output, 'DETERMINISTIC', failing, tagged_as_stochastic
            else:
                print xname, operator, operation, output, 'RANDOMISED', failing, tagged_as_stochastic
                if tooverbose:
                    for value, voodf in oodf.groupby('value'):
                        print '\t', value, map(str, voodf['host'].unique())
    stochastic_failing(df)


FAILING_AFTER_CELL_NASTINESS = {
    'Fails with sine': [HCTSAOperations.MF_GP_hyperparameters_covSEiso_covNoise_1_200_first[2]],
    'Fails with noisysinusoid': [
        HCTSAOperations.NL_TISEAN_d2_1_10_0[2],
        HCTSAOperations.NL_TISEAN_d2_ac_10_001[2],
    ]
}


def test_one(operation, engine='matlab'):
    engine = PyopyEngines.engine_or_matlab_or_octave(engine)
    hctsa_prepare_engine(engine)
    print operation.what().id()
    for name, x in (('noise', hctsa_noise()),
                    ('sine', hctsa_sine()),
                    ('noisysinusoid', hctsa_noisysinusoid()),
                    ('randn10000', np.random.RandomState(0).randn(10000))):
        print '-' * 80
        print name
        x = engine.put('x', hctsa_prepare_input(x, z_scored=True))
        try:
            operation.transform(x, eng=engine)
        except Exception as ex:
            print ex


if __name__ == '__main__':

    from joblib import Parallel, delayed
    n_jobs = 4
    Parallel(n_jobs=n_jobs)(delayed(check_benchmark_bindings)(x=xfact(),
                                                              xname=xname,
                                                              extra='pycharm',
                                                              n_jobs=n_jobs)
                            for (xname, xfact), _ in product(TS_FACTORIES.items(), range(4)))
