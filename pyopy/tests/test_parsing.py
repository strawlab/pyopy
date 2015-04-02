# coding=utf-8
from pyopy.base import MatlabSequence
from pyopy.code import parse_matlab_params


def test_parse_matlab_params():

    # Numeric cells must be properlly marked - otherwise they get translated as double and cannot be accessed by {}
    assert parse_matlab_params("-1,0.05,0.05,{1,8}") == [-1, 0.05, 0.05, (1.0, 8.0, '_celltrick_')]

    assert parse_matlab_params("-1,0.05,0.05,{1,8,{2}}") == [-1, 0.05, 0.05, (1.0, 8.0, (2.0, '_celltrick_'))]

    # Matlab sequences must be properlly handled
    assert parse_matlab_params("1:3") == [MatlabSequence('1:3')]

    # This is correct behavior, the MatlabSequence list must be properly flattened a posteriori...
    assert parse_matlab_params("[1:3]") == [[MatlabSequence('1:3')]]

    assert parse_matlab_params("[2,4,6,8,2,4,6,8],[0,0,0,0,1,1,1,1]") == \
        [(2, 4, 6, 8, 2, 4, 6, 8), (0, 0, 0, 0, 1, 1, 1, 1)]

    assert parse_matlab_params("{'covSum',{'covSEiso','covNoise'}},1,200,'resample', 1:3") == \
        [('covSum', ('covSEiso', 'covNoise')), 1, 200, 'resample', MatlabSequence('1:3')]

    assert parse_matlab_params(
        "{'covSum',{'''R'',2,''M'',1,''P'',2,''Q'',1,''M''','covNoise'}},1,200,'resample', 1:3") == \
        [('covSum', ("'R',2,'M',1,'P',2,'Q',1,'M'", 'covNoise')), 1, 200, 'resample', MatlabSequence('1:3')]

    assert parse_matlab_params("'ar','''R'',2,''M'',1,''P'',2,''Q'',1,''M'''") == \
        ['ar', "'R',2,'M',1,'P',2,'Q',1,'M'"]
