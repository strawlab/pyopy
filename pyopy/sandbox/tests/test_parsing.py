# coding=utf-8
import numpy as np
from numpy.testing.utils import assert_array_equal

from pyopy.base import MatlabSequence
from pyopy.sandbox.parsing import parse_matlab_function, MatlabId


def test_basic_parsing():
    # No params
    name, params = parse_matlab_function('A()')
    assert name == MatlabId('A')
    assert params == []


def test_strings_parsing():
    # Strings
    name, params = parse_matlab_function("A(y, 'y', '')")
    assert name == MatlabId('A')
    assert params == [MatlabId('y'), 'y', '']

    # Strings with quote escaping
    name, params = parse_matlab_function("A('ar','''R'',2,''M'',1,''P'',2,''Q'',1')")
    assert name == MatlabId('A')
    assert params == ['ar', "'R',2,'M',1,'P',2,'Q',1"]


def test_numbers_parsing():
    # Numbers
    name, params = parse_matlab_function("A(y, 1, 1.2, 1e32, inf, -inf, Inf, -Inf)")
    assert name == MatlabId('A')
    assert params == [MatlabId('y'), 1, 1.2, 1e32, np.inf, -np.inf, np.inf, -np.inf]

    # nans
    name, params = parse_matlab_function("A(nan, NaN)")
    assert name == MatlabId('A')
    assert len(params) == 2
    assert np.isnan(params[0])
    assert np.isnan(params[1])


def test_sequence_parsing():
    # Basic
    name, params = parse_matlab_function('A(0:5)')
    assert name == MatlabId('A')
    assert params == [MatlabSequence('0:5')]

    # Steps with reals involved
    name, params = parse_matlab_function('A(0:0.05:0.95)')
    assert name == MatlabId('A')
    assert params == [MatlabSequence('0:0.05:0.95')]


def _test_matrix_parsing():
    # Empty matrix
    name, params = parse_matlab_function("A([])")
    assert name == MatlabId('A')
    assert len(params) == 1
    assert_array_equal(params[0], np.array([]))

    # Singleton matrix
    name, params = parse_matlab_function("A([1])")
    assert name == MatlabId('A')
    assert params == [1]

    name, params = parse_matlab_function("A([[[1]]])")
    assert name == MatlabId('A')
    assert params == [1]

    name, params = parse_matlab_function("A([[[1]]; 2])")
    assert name == MatlabId('A')
    assert_array_equal(params[0], np.array([[1], [2]]))


def test_matrix_parsing():
    name, params = parse_matlab_function("A([[[[1]]] 2; 3 4])")
    assert name == MatlabId('A')
    assert_array_equal(params[0], np.array([[1, 2], [3, 4]]))

    # also: [[2 3] 4] -> [2 3 4]
    # see octave isscalar, and how octave interprets [[[0]]]...



def __test_matrix_parsing():
    # Mixed slice, constant row matrix
    name, params = parse_matlab_function("A([1, 2:8])")
    assert name == MatlabId('A')
    assert len(params) == 1
    assert_array_equal(params[0], np.arange(1, 9).reshape(1, -1))

    # Column matrix
    name, params = parse_matlab_function("A([1; 2; 3])")
    assert name == MatlabId('A')
    assert len(params) == 1
    assert_array_equal(params[0], np.array([[1], [2], [3]]))

    # Mixed slice, constant 2D matrix
    name, params = parse_matlab_function("A([1, 2:3; 3:5])")
    assert name == MatlabId('A')
    assert len(params) == 1
    assert_array_equal(params[0], np.array([[1, 2, 3], [3, 4, 5]]))


def test_cell_parsing():

    # Only numeric cells
    name, (cell,) = parse_matlab_function('A({1, 2, 3})')
    assert name == MatlabId('A')
    assert_array_equal(cell, np.array([1, 2, 3], dtype=object))

    # Nested cells
    name, (cell,) = parse_matlab_function("A({'covSum',{'covSEiso','covNoise'}})")
    assert name == MatlabId('A')
    assert cell.ndim == 1
    assert len(cell) == 2
    assert cell[0] == 'covSum'
    assert_array_equal(cell[1], np.array(['covSEiso', 'covNoise'], dtype=object))

    # 2D cells
    # name, (cell,) = parse_matlab_function("A({1 2; {'covSEiso'}, 'a'})")
    # assert name == MatlabId('A')
    #


def test_hctsa_examples():
    pass
