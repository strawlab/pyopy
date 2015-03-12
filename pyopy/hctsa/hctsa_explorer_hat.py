# coding=utf-8
from operator import itemgetter
import time

import numpy as np

from pyopy.base import EngineException, PyopyEngines
from pyopy.hctsa.hctsa_bindings import CO_AddNoise, SY_SpreadRandomLocal, HCTSAOperations
from pyopy.hctsa.hctsa_data import hctsa_sine
from pyopy.hctsa.hctsa_setup import prepare_engine_for_hctsa


# ncs = computable_econometrics()
# oks = []
# fails = []
# with PyMatBridgeEngine() as eng:
#     prepare_engine_for_hctsa(eng)
#     # set_eng(fex, eng)  # This should return a new copy, the list of features should indeed return a new copy too
#     # set_eng(fexc, eng)  # This should return a new copy, the list of features should indeed return a new copy too
#     x = hctsa_sine()
#     x = check_prepare_hctsa_input(x)
#     for name, fex in ncs:
#         print name
#         set_eng(fex, eng)
#         # Of course, this does not work if x is already in matlab land...
#         try:
#             res = fex.transform(x)
#             oks.append(name)
#         except:
#             fails.append(name)
#
# print '\n'.join(fails)
# print '%d ok, %d failed' % (len(oks), len(fails))
#
# exit(33)


def compare(eng,
            x=hctsa_sine(),
            fex=CO_AddNoise(),
            timeout=None):
    try:
        start = time.time()
        result = fex.transform(x, eng=eng)
        return result, 'Success', time.time() - start
    except EngineException, e:
        return None, e.engine_response.stdout, e.engine_response.code, None
    except Exception, e:
        return None, str(e), None, None

# eng = PyopyEngines.octave()
eng = PyopyEngines.matlab()
print 'Preparing...'
prepare_engine_for_hctsa(eng)
print 'Comparing...'
# TODO: reshape should be done automatically by HCTSA classes, but can be bad for performance
#       or use oned_as='col'...
x = eng.put('x', hctsa_sine().reshape(-1, 1))
print compare(eng, fex=HCTSAOperations.CO_CompareMinAMI_std2_2_80[1], x=x)
print compare(eng, fex=HCTSAOperations.EN_PermEn_2[1], x=x)
print compare(eng, fex=HCTSAOperations.CO_HistogramAMI_1_even_10[1], x=x)
print compare(eng, fex=HCTSAOperations.PP_Compare_medianf2[1], x=x)
print compare(eng, fex=HCTSAOperations.MF_CompareAR_1_10_05[1], x=x)
print compare(eng, fex=HCTSAOperations.WL_cwt_db3_32[1], x=x)
exit(22)


def arrays2cells_and_partial(eng='octave'):

    eng = PyopyEngines.engine_or_matlab_or_octave(eng)

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
        result.append(SY_SpreadRandomLocal(l='ac5').eval(eng, array))
        taken += time.time() - start
    print result
    print 'python-land loop took %.2f seconds' % taken
    # python-land loop took 6.18 seconds (octave)
    # python-land loop took 1.27 seconds (matlab) - bye oct2py call overhead + octave slower (use faster builds)


#
# Assumptions:
#   - Only operators with more than one return in categories can be return-type variant
#   - Anything returned is either a single value or a dict subclass
#   - Output fields depend only on input parameters, except for the time-series.
#     That is, the same parameters will always lead to the same output fields.
#


def guess_hctsa_outputs(hctsaer, eng, data=hctsa_sine()):
    def process_hctsa_output(out, keep_nans=False):
        if not isinstance(out, dict):
            return None, out
        if keep_nans:
            return sorted(out.items())
        return sorted((k, v) for k, v in out.iteritems() if not np.isnan(v))
    out = process_hctsa_output(hctsaer.eval(eng, data))
    if isinstance(out, list):
        return hctsaer.what().id(), map(itemgetter(0), out)
    return hctsaer.what().id(), None

# data = hctsa_sine()
# with PyMatBridgeEngine() as eng:
#     prepare_engine_for_hctsa(eng)
#     hctsaer = CO_TranslateShape()
#     hctsaer = HCTSA_Categories.MS_shannon_2_1t10
#     print guess_hctsa_outputs(hctsaer, eng)
#     # print guess_hctsa_outputs(hctsaer)
#     # print HCTSA_CO_TranslateShape(eng, data)
#     # print process_hctsa_output(HCTSA_CO_TranslateShape(eng, data))


# MATLAB_ENGINES = (
#     ('oct2py',      'octave', Oct2PyEngine()),
#     ('pymatbridge', 'octave', PyMatBridgeEngine(octave=True)),
#     ('pymatbridge', 'matlab', PyMatBridgeEngine(octave=False)),
# )
#
# with ExitStack() as stack:
#     for name, octave, eng in MATLAB_ENGINES:
#         print name
#         stack.enter_context(eng)
#         eng.warmup()
#         prepare_engine_for_hctsa(eng)


# oks = []
# failed = []
# # print HCTSA_Categories.all()
# # with PyMatBridgeEngine(octave=True) as eng:
# with Oct2PyEngine() as o2eeng, PyMatBridgeEngine(octave=True) as pmbo, PyMatBridgeEngine() as pmbm:
#     prepare_engine_for_hctsa(o2eeng)
#     prepare_engine_for_hctsa(pmbo)
#     prepare_engine_for_hctsa(pmbm)
#     x = (x - x.mean()) / x.std(ddof=1)
#     for fname, fex in HCTSA_Categories.all():
#         try:
#             print fname, fex.configuration().id(), fex.output_names(eng, x=x)
#             oks.append(fex)
#         except:
#             print fex.configuration().id(), 'FAILED'
#             failed.append(fex)
# print '-' * 80
# for fex in failed:
#     print fex.configuration().id()
# print len(oks), len(failed)
#
#
# class FeatureInfo(object):
#     def __init__(self, outputs, timings):
#         super(FeatureInfo, self).__init__()
#         self.outputs = outputs
#         self.timings = timings


# Matlab stores data in column major format, for loops iterate over columns

if __name__ == '__main__':

    # arrays2cells_and_partial(eng='matlab')
    # arrays2cells_and_partial(eng='octave')
    # hctsa_partials_poc()
    pass
