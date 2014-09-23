# coding=utf-8
import time

import numpy as np
from oscail.common.config import Configurable

from pyopy.hctsa.bindings.hctsa_bindings import HCTSA_SY_SpreadRandomLocal, HCTSA_CO_AddNoise
from pyopy.hctsa.hctsa_utils import prepare_engine_for_hctsa
from pyopy.matlab_utils import Oct2PyEngine, py2matstr, PyMatBridgeEngine
import inspect


class CO_AddNoise(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Analyzes changes in the automutual information function with the addition of
    % noise to the input time series.
    % Adds Gaussian-distributed noise to the time series with increasing standard
    % deviation, eta, across the range eta = 0, 0.1, ..., 2, and measures the
    % mutual information at each point using histograms with
    % nbins bins (implemented using CO_HistogramAMI).
    %
    % The output is a set of statistics on the resulting set of automutual
    % information estimates, including a fit to an exponential decay, since the
    % automutual information decreases with the added white noise.
    %
    % Can calculate these statistics for time delays 'tau', and for a number 'nbins'
    % bins.
    %
    % This algorithm is quite different, but was based on the idea of 'noise
    % titration' presented in: "Titration of chaos with added noise", Chi-Sang Poon
    % and Mauricio Barahona P. Natl. Acad. Sci. USA, 98(13) 7107 (2001)
    %
    ----------------------------------------
    """

    _partials_cache = {}
    _num_partials = 0

    outnames = ('ac1',
                'ac2',
                'fitexpa',
                'fitexpadjr2',
                'fitexpb',
                'fitexpr2',
                'fitexprmse',
                'fitlina',
                'fitlinb',
                'meanch',
                'mse',
                'pcrossmean',
                'pdec')

    def __init__(self, tau=1, meth='quantiles', nbins=20):
        super(CO_AddNoise, self).__init__(add_descriptors=False)
        self.tau = tau
        self.meth = meth
        self.nbins = nbins

    def eval(self, engine, x):
        return HCTSA_CO_AddNoise(engine,
                                 x,
                                 tau=self.tau,
                                 meth=self.meth,
                                 nbins=self.nbins)

    def as_partial_call(self, engine=None):
        if self.configuration().id() not in self._partials_cache:
            func_name = self.__class__.__name__
            parameters_sorted_as_in_matlab_call = inspect.getargspec(self.__init__)[0][1:]
            parameters = []
            for param in parameters_sorted_as_in_matlab_call:
                val = self.__getattribute__(param)
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

partials = (
    CO_AddNoise().as_partial_call(),
    CO_AddNoise(meth=None).as_partial_call(),
    CO_AddNoise(tau=1.2, meth=None).as_partial_call()
)


def arrays2cells_and_partial():

    engine = Oct2PyEngine()

    # Prepare for HCTSA
    prepare_engine_for_hctsa(engine)

    # Generate data
    arrays = [np.random.randn(size) for size in (100, 500, 100, 35, 200, 130, 230)]
    # To octave land
    _ = engine.put('x', [arrays])  # N.B. needs to be list to make a cell

    # cellfun passing lambda (can be useful for partial application and to aggregate operators)
    start = time.time()
    engine.run_command('f1=@(x) SY_SpreadRandomLocal(x, l=\'ac5\')')
    print engine.run_command('cellfun(f1, x, \'UniformOutput\', 0)', outs2py=True)
    print 'cellfun with partial took %.2f seconds' % (time.time() - start)
    # cellfun with partial took 5.42 seconds

    # cellfun without lambda
    start = time.time()
    print engine.run_command('cellfun(\'SY_SpreadRandomLocal\', x, \'UniformOutput\', 0)', outs2py=True)
    print 'cellfun took %.2f seconds' % (time.time() - start)
    # cellfun took 43.95 seconds

    # python-land loop
    taken = 0
    for array in arrays:
        array = engine.put('blah', array)
        start = time.time()
        HCTSA_SY_SpreadRandomLocal(engine, array)  # even without memory copy hurdles
                                                   # add 0.1s per call because of python introsprection
                                                   # and bytecode disassembly
        taken += time.time() - start
    print 'python-land loop took %.2f seconds' % taken
    # python-land loop took 6.18 seconds


if __name__ == '__main__':

    arrays2cells_and_partial()