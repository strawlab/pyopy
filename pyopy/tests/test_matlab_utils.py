# coding=utf-8
from pyopy.matlab_utils import Oct2PyEngine


def test_oct2py_engine():
    engine = Oct2PyEngine()
    val = engine.put('a', 12)
    assert val.get() == 12