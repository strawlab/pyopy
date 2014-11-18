# coding=utf-8
import numpy as np
from whatami import whatable


# ----- Input preprocessing

def matlab_standardize(x):
    """Standardize x so that they are z-scores (as could have been done in matlab-land).
    Note that default numpy and matlab std computation differ:
      See http://stackoverflow.com/questions/7482205/precision-why-do-matlab-and-python-numpy-give-so-different-outputs
    """
    return (x - x.mean()) / (x.std(ddof=1))


def check_prepare_hctsa_input(x, z_scored=False):
    """
    Given a 1D array x, prepare it to be transferred to hctsa land.
      - HCTSA expects floating point numbers.
        So we cast if needed.
      - HCTSA expects column vectors.
        In order for this to work regardless of oned_as configuration of the engine, we reshape it to (-1, 1).
    N.B. other preconditions for some HCTSA operators, like z-scored time series, must be hadled somewhere else.
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)  # hctsa expects a column vector
    elif not 2 == x.ndim:
        raise Exception('Only one dimensional column vectors for HCTSA, please')
    elif x.shape[1] != 1:
        raise Exception('Only column vectors for HCTSA, please')
    if z_scored:
        return matlab_standardize(x.astype(np.float))
    return x.astype(np.float)


# ----- Transformers, bare minimum until we get the final abstractions from oscail

@whatable
class Chain(object):

    def __init__(self, transformers=()):
        super(Chain, self).__init__()
        self.chain = transformers

    def transform(self, x):
        for transformer in self.chain:
            x = transformer.transform(x)
        return x


@whatable
class MatlabStandardize(object):

    def __init__(self):
        super(MatlabStandardize, self).__init__()

    @staticmethod
    def transform(x):
        return matlab_standardize(x)