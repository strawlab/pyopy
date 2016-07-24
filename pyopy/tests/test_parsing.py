# coding=utf-8
import numpy as np
from numpy.testing.utils import assert_array_equal

from pyopy.base import MatlabSequence, MatlabId
from pyopy.translation import parse_matlab_function


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
    assert params[0] != params[0]
    assert params[1] != params[1]


def test_sequence_parsing():
    # Basic
    name, params = parse_matlab_function('A(0:5)')
    assert name == MatlabId('A')
    assert params == [MatlabSequence('0:5')]

    # Steps with reals involved
    name, params = parse_matlab_function('A(0:0.05:0.95)')
    assert name == MatlabId('A')
    assert params == [MatlabSequence('0:0.05:0.95')]


def test_matrix_parsing():
    # Empty matrix -> empty array
    name, (matrix,) = parse_matlab_function("A([])")
    assert name == MatlabId('A')
    assert_array_equal(matrix, np.array([]))

    # Single element matrix -> python scalar
    name, (matrix,) = parse_matlab_function("A([1])")
    assert name == MatlabId('A')
    assert matrix == 1

    name, (matrix,) = parse_matlab_function("A([[[1]]])")
    assert name == MatlabId('A')
    assert matrix == 1

    # Single row matrix -> 1D array
    name, (matrix,) = parse_matlab_function("A([[[1 2]]])")
    assert name == MatlabId('A')
    assert_array_equal(matrix, np.array([1, 2]))

    name, (matrix,) = parse_matlab_function("A([[[1 2]], 3 4 5:6, 7:8])")
    assert name == MatlabId('A')
    assert_array_equal(matrix, np.array([1, 2, 3, 4, 5, 6, 7, 8]))

    # Many rows matrices -> 2D array
    name, (matrix,) = parse_matlab_function("A([1; 2])")
    assert name == MatlabId('A')
    assert_array_equal(matrix, np.array([[1],
                                         [2]]))

    name, (matrix,) = parse_matlab_function("A([[[1]] 2:4; 1, 2, [3 4]])")
    assert name == MatlabId('A')
    assert_array_equal(matrix, np.array([[1, 2, 3, 4],
                                         [1, 2, 3, 4]]))

    # More freeform matrix construction: 2D matrices concatenation
    name, (matrix,) = parse_matlab_function("A([[1;2] [3;4]])")
    assert name == MatlabId('A')
    assert_array_equal(matrix, np.array([[1, 3],
                                         [2, 4]]))


def test_cell_parsing():

    # An empty cell
    name, (cell,) = parse_matlab_function('A({})')
    assert name == MatlabId('A')
    assert_array_equal(cell, np.array([], dtype=object))

    # Only numeric cell arrays
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
    name, (cell,) = parse_matlab_function("A({1 2; {'covSEiso'}, 'a'})")
    assert name == MatlabId('A')
    assert cell.ndim == 2
    assert len(cell) == 2
    assert_array_equal(cell[0], np.array([1, 2], dtype=object))
    assert_array_equal(cell[1], np.array([np.array('covSEiso', dtype=object), 'a']))
