# coding=utf-8
import json
import os.path as op

import numpy as np
import time


# Write OK and Failed tuples
from pyopy.matlab_utils import PyMatlabEngine


def check_bindings(engine=None,
                   operations=None,
                   forbidden=(('HCTSA_MF_ARMA_orders', 'Enters Oct2Py interact mode'),
                              ('HCTSA_SY_DriftingMean', 'Has a bug (l is not defined) and enters Oct2Py interact mode'),
                              ('HCTSA_TSTL_predict', 'Takes too long?',)),
                   sizes=(100, 1000, 10000),
                   ts_factory=lambda size: np.random.RandomState(0).randn(size),
                   dest_file=op.join(op.dirname(__file__), 'hctsa_octave_checks.py'),
                   error_extractor=lambda e: str(e).rpartition("Octave returned:")[2].strip()):

    if engine is None or engine == 'octave':
        from pyopy.hctsa.hctsa_utils import add_hctsa_path_to_engine
        from pyopy.matlab_utils import Oct2PyEngine
        engine = Oct2PyEngine()
        add_hctsa_path_to_engine(engine)
    elif engine == 'matlab':
        from pyopy.hctsa.hctsa_utils import add_hctsa_path_to_engine
        from pyopy.matlab_utils import PyMatlabEngine
        engine = PyMatlabEngine()
        add_hctsa_path_to_engine(engine)

    # Some operations that make the entire experiment fail
    forbidden = dict(forbidden)

    for size in sizes:
        print 'Time series size: %d' % size
        oks = []
        failed = []
        x = engine.put('x', ts_factory(size))
        if operations is None:
            from hctsa_bindings import ALL_OPS as operations
        for operation in operations:
            funcname = operation.__name__
            print funcname
            if funcname in forbidden:
                failed.append((funcname, forbidden[funcname]))
                continue
            try:
                start = time.time()
                operation(engine, x)  # Lame success test
                oks.append((funcname, time.time() - start))
            except Exception, e:
                failed.append((funcname, error_extractor(e)))

        with open(dest_file, 'a') as writer:
            writer.write('\n\nOK_OPS_%d = [\n    %s]' % (size, '    '.join(['(\'%s\', %g),\n' % ok for ok in oks])))
            writer.write('\n\n\nFAILED_OPS_%d = [\n    %s]' % (size,
                         '    '.join(['("%s", %s),\n' % (funcname, json.dumps(reason))
                                      for funcname, reason in failed])))

if __name__ == '__main__':
    check_bindings()
    # check_bindings(
    #     engine='matlab',
    #     forbidden=(),
    #     dest_file=op.join(op.dirname(__file__), 'hctsa_matlab_checks.py'),
    #     error_extractor=lambda e: str(e).strip()
    # )

#
# Other notes
#   - SY_StdNthDer: very simple, but not well implemented
#
