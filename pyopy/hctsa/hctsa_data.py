# coding=utf-8
"""HCTSA test time series."""
import os.path as op
import numpy as np
from pyopy.hctsa.hctsa_config import HCTSA_TESTTS_DIR


def hctsa_sine():
    return np.loadtxt(op.join(HCTSA_TESTTS_DIR, 'SY_sine.dat'))


def hctsa_noise():
    return np.loadtxt(op.join(HCTSA_TESTTS_DIR, 'SY_noise.dat'))


def hctsa_noisysinusoid():
    return np.loadtxt(op.join(HCTSA_TESTTS_DIR, 'SY_noisysinusoid.dat'))


HCTSA_TEST_TIME_SERIES = (
    ('sine', hctsa_sine),
    ('noise', hctsa_noise),
    ('noisysinusoid', hctsa_noisysinusoid),
)
