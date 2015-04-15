# coding=utf-8
"""High lever API for the library."""
import copy


def prepare(engine='matlab',
            warmup=True,
            prepare_engine=True,
            prepare_operators=True,
            eng=[None],
            operations_with_eng=[None]):

    # TODO: a fast function to check if hctsa has been initialised
    if eng[0] is None:
        from pyopy.base import PyopyEngines
        from hctsa_install import hctsa_prepare_engine
        print 'Starting engine'
        eng[0] = PyopyEngines.engine_or_matlab_or_octave(engine)
        if warmup:
            print 'Warming up'
            eng[0].warmup()
        if prepare_engine:
            print 'Configuring HCTSA'
            hctsa_prepare_engine(eng[0])
        if prepare_operators:
            print 'Setting up HCTSA operators'
            operations_with_eng[0] = hctsa_operations_use_eng(eng=eng[0])
        print 'Hooray, we can use HCTSA now...'
    return eng[0], operations_with_eng[0]


try:
    import hctsa_bindings as bindings
except ImportError:
    bindings = None

try:
    from hctsa_bindings import HCTSAOperations as operations
except ImportError:
    operations = None

try:
    from hctsa_catalog import HCTSACatalog as catalog
    catalog = catalog.catalog()
except ImportError:
    catalog = None


def hctsa_operations_use_eng(operators=None, eng=None, inplace=True):
    if eng is None:
        eng = prepare()
    if operators is None:
        operators = catalog.allops()
    operators_copy = []
    for name, operator in operators:
        if not inplace:
            operator = copy.copy(operator)
        operator.use_eng(eng)
        operators_copy.append((name, operator))
    return operators_copy
