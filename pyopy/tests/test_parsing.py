# coding=utf-8
from pyopy.base import MatlabSequence
from pyopy.code import parse_matlab_params


def test_parse_matlab_params():

    # the MatlabSequence must be properly flattened a posteriori...
    assert parse_matlab_params("[1:3]") == [[MatlabSequence('1:3')]]

    assert parse_matlab_params("1:3") == [MatlabSequence('1:3')]

    assert parse_matlab_params("[2,4,6,8,2,4,6,8],[0,0,0,0,1,1,1,1]") == \
        [(2, 4, 6, 8, 2, 4, 6, 8), (0, 0, 0, 0, 1, 1, 1, 1)]

    assert parse_matlab_params("{'covSum',{'covSEiso','covNoise'}},1,200,'resample', 1:3") == \
        [('covSum', ('covSEiso', 'covNoise')), 1, 200, 'resample', MatlabSequence('1:3')]

    assert parse_matlab_params(
        "{'covSum',{'''R'',2,''M'',1,''P'',2,''Q'',1,''M''','covNoise'}},1,200,'resample', 1:3") == \
        [('covSum', ("'R',2,'M',1,'P',2,'Q',1,'M'", 'covNoise')), 1, 200, 'resample', MatlabSequence('1:3')]

    assert parse_matlab_params("'ar','''R'',2,''M'',1,''P'',2,''Q'',1,''M'''") == ['ar', "'R',2,'M',1,'P',2,'Q',1,'M'"]
