# coding=utf-8
"""Matlab source code parsing and manipulation.
These functions are very ad-hoc, but work for all our cases.
"""
from functools import partial
import re
import os.path as op

import numpy as np

from pyopy.externals.ompc.ompcply import translate_to_str


def matlab_funcname_from_filename(mfile):
    """Returns the name of the mfile.
    Matlab main functions must be named after the file they reside in.
    """
    return op.splitext(op.basename(mfile))[0]


def parse_matlab_funcdef(mfile,
                         funcdef_pattern=re.compile(r'^function\s*(\S.+)\s*=\s*(\S+)\s*\((.*)\)$',
                                                    flags=re.MULTILINE)):
    """Splits a matlab function file into components.

    Parameters
    ----------
    mfile: string
        The path to the m-function-file to parse

    funcdef_pattern: regular expression pattern
        A pattern matcher that can split the contents of the file into the output components

    Returns
    -------
    a tuple doc (string), out (string), funcname (string), parameters (string list), code (string)
    """
    expected_func_name = op.splitext(op.basename(mfile))[0]
    with open(mfile) as reader:
        text = reader.read()
        doc, out, funcname, parameters, code = funcdef_pattern.split(text, maxsplit=1)
        if not funcname == expected_func_name:
            raise Exception('Problem parsing %s.\n'
                            'The function name does not correspond to the file name' %
                            mfile)
        parameters = map(str.strip, parameters.split(','))
        return doc, out, funcname, parameters, code


def infer_default_values(parameters, code):
    """Very brittle attempt at inferring parameters default values for matlab functions.
    At the moment we assume they are set by a "if nargin < blah" idiom.

    Parameters
    ----------
    parameters: string list
        A list of the parameters of the function,

    code: string
        The code of the function.

    Returns
    -------
    A string list with the default value inferred for each parameter (empty strings for unknown).
    """

    default_values = []
    for i, param in enumerate(parameters):
        val = code.partition('if nargin < %d' % (i + 2))[2].partition('%s = ' % param)[2].partition(';')[0]
        default_values.append(val.strip())
    return default_values


def rename_matlab_func(mfile, old_wrong_name):
    """Renames inplace a function inside an mfile, given a known "old wrong name".

    We essentially just search and replace old_wrong_name with the name of the m-file.

    Parameters
    ----------
    mfile: string
        The path to the mfile.

    old_wrong_name: string
        The string we will substitute in the file

    Returns
    -------
    Nothing, but beware that the mfile is potentially changed.
    """
    with open(mfile) as reader:
        text = reader.read()
    with open(mfile, 'w') as writer:
        writer.write(text.replace(old_wrong_name, matlab_funcname_from_filename(mfile)))


def parse_matlab_params(matlab_params_string, int2float=True):
    """Parses a matlab parameter values string and returns each value in python land.

    For parsing matlab expressions we use GPLed OMPC
    http://ompc.juricap.com/

    Parameters
    ----------
    params_string: string
        A string containing the parameters for a call to matlab, for example:
          "0.5,'best'"
        or the more exotic but also real:
          "'ar','''R'',2,''M'',1,''P'',2,''Q'',1'"

    Returns
    -------
    A list with the parameter values, in python land

    Examples
    --------
    >>> parse_matlab_params("[2,4,6,8,2,4,6,8],[0,0,0,0,1,1,1,1]")
    [(2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0), (0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0)]
    >>> parse_matlab_params("{'covSum',{'covSEiso','covNoise'}},1,200,'resample'")
    [('covSum', ('covSEiso', 'covNoise')), 1, 200, 'resample']
    >>> parse_matlab_params("0.5,'best'")
    [0.5, 'best']
    >>> parse_matlab_params("'ar','''R'',2,''M'',1,''P'',2,''Q'',1'")
    ['ar', "'R',2,'M',1,'P',2,'Q',1"]
    """
    def ompc2evaluable(ompc_string):

        def ompc2py(opmc_string,
                    prefix='mstring(',
                    dopen='(', dclose=')',
                    gfunc=lambda x: x):
            def find_balanced_delim(string, dopen='(', dclose=')'):
                num_open = 1
                for i, c in enumerate(string):
                    if c == dopen:
                        num_open += 1
                    if c == dclose:
                        num_open -= 1
                    if num_open == 0:
                        return i
            mstring1 = opmc_string.find(prefix)
            if mstring1 < 0:
                return opmc_string
            mstring2 = mstring1 + len(prefix)
            mstring3 = mstring2 + find_balanced_delim(opmc_string[mstring2:], dopen=dopen, dclose=dclose)
            return opmc_string[0:mstring1] + gfunc(opmc_string[mstring2:mstring3]) + opmc_string[mstring3+1:]

        def flatten(opmc_string, flatter):
            old = opmc_string
            new = flatter(old)
            while old != new:
                old = new
                new = flatter(old)
            return new

        mstring2py = partial(ompc2py,
                             prefix='mstring(',
                             gfunc=lambda x: '"%s"' % x[1:-1])
        mcat2py = partial(ompc2py, prefix='mcat(')
        mslice2py = partial(ompc2py,
                            prefix='mslice[', dopen='[', dclose=']',
                            gfunc=lambda x: 'MatlabSequence("%s")' % x)

        for flatter in (mstring2py, mslice2py, mcat2py):
            ompc_string = flatten(ompc_string, flatter)

        # Evaluate to get the matlab value
        from pyopy.base import MatlabSequence  # need this in scope for eval
        pyval = eval(ompc_string)   # literal_eval does not cover MatlabSequence
        if int2float:
            try:
                return tuple(np.array(pyval, dtype=np.float))
                # N.B. numeric scalars also finally get converted to double (casted by numpy, unpacked by tuple)
                # this is too clever, better make explicit
            except:
                pass
        return pyval

    if not matlab_params_string:
        return []
    return map(ompc2evaluable, translate_to_str(matlab_params_string).splitlines())


def outputs_from_command(command):
    """Very ad-hoc function that finds out the names of the outputs of a matlab command.

    It surely won't cover all of matlab statement call syntax, so complete with new cases as they come by.

    Examples
    --------

    >>> outputs_from_command('clear x')
    []
    >>> outputs_from_command('ones(1);')
    ['ans']
    >>> outputs_from_command('[x, y] = meshgrid(5, 5);')
    ['x', 'y']
    """
    # Is it a function call? - not robust at all
    is_function_call = '=' in command or '(' in command and ')' in command
    if not is_function_call:
        return []
    has_out_names = '=' in command[:command.find('(')]
    if not has_out_names:
        return ['ans']
    out_names = command[:command.find('(')].partition('=')[0].strip().replace('[', '').replace(']', '')
    return map(str.strip if not isinstance(command, unicode) else unicode.strip, out_names.split(','))


#####################################################################
#
# Old parameter-line parsing machinery (pre OMPC reimplementation)
#
#####################################################################
#
#
# def matlab_val_to_python_val(matlabval):
#     """Given a string representing a matlab value, create an equivalent python value.
#
#     Parameters
#     ----------
#     matlabval: string
#         A string representing a matlab value (e.g. "[1, 2, 3]", "1.1" or "1:3")
#
#     Returns
#     -------
#     The value in python-land.
#
#     Examples
#     --------
#     >>> matlab_val_to_python_val('1.1')
#     1.1
#     >>> matlab_val_to_python_val('[1, 2, 3]')
#     [1, 2, 3]
#     >>> matlab_val_to_python_val('[1:2:3]')
#     MatlabSequence('1:2:3')
#     >>> matlab_val_to_python_val('\\'max\\'')
#     'max'
#     """
#     try:
#         return MatlabSequence(matlabval)
#     except:
#         return eval(matlabval)  # FIXME... code injections...
#
#
# def matlab_triple_quotes_to_python(matlabval):
#     """Replaces all occurrences of matlab-triple-quotes by the python equivalent double-quotes-escape-single-quotes.
#
#     From http://www.ee.columbia.edu/~marios/matlab/matlab_tricks.html
#       "Info about the dreaded triple quotes ''' in eval(). A really nasty way to disable
#       quote's special functionality. In other languages one would expect to escape
#       the quote by \' but not in Matlab. What you really need to know is that:
#       >> eval('disp(''''''This is a string'''''')')
#       'This is a string'"
#
#     Parameters
#     ----------
#
#     matlabval: string
#         A string representing a matlab value (also things like comma separated values are handled properly)
#
#     Examples
#     --------
#     >>> print matlab_triple_quotes_to_python("'ar','''R'',2,''M'',1,''P'',2,''Q'',1'")
#     'ar',"'R',2,'M',1,'P',2,'Q',1"
#
#     >>> print matlab_triple_quotes_to_python("'''R'',2,''M'',1,''P'',2,''Q'',1'")
#     "'R',2,'M',1,'P',2,'Q',1"
#     """
#     while "'''" in matlabval:
#         pre, _, post = matlabval.partition("'''")
#         post = '"\'' + post.\
#             replace("''", '33matlabscapedquotesnastiness4$'). \
#             replace("'", '"'). \
#             replace('33matlabscapedquotesnastiness4$', "'")  # Usefully nasty
#         matlabval = pre + post
#     return matlabval
#
# def parse_matlab_params(params_string, use_ompc=True):
#     """Parses a matlab parameter values string and return each value in python land.
#
#     Parameters
#     ----------
#     params_string: string
#         A string containing the parameters for a call to matlab, for example:
#           "0.5,'best'"
#         or the more exotic but also real:
#           "'ar','''R'',2,''M'',1,''P'',2,''Q'',1'"
#
#     Returns
#     -------
#     A list with the parameter values, in python land
#
#     Examples
#     --------
#     >>> parse_matlab_params("[2,4,6,8,2,4,6,8],[0,0,0,0,1,1,1,1]")
#     [[2,4,6,8,2,4,6,8], [0,0,0,0,1,1,1,1]]
#     # >>> parse_matlab_params("{'covSum',{'covSEiso','covNoise'}},1,200,'resample'")
#     # This fails so I go to a full parser (ompc)
#     >>> parse_matlab_params("0.5,'best'")
#     [0.5, 'best']
#     >>> parse_matlab_params("'ar','''R'',2,''M'',1,''P'',2,''Q'',1'")
#     ['ar', "'R',2,'M',1,'P',2,'Q',1"]
#     """
#     if use_ompc:
#         return parse_matlab_params_ompc(params_string)
#     #
#     # A note on parsing hardness:
#     #   - csv does not support adding different opening and closing quote characters
#     #   - of course regexps per se are useless for balanced delimiters matching,
#     # So if we detect nested structures we will need to go for a full parser
#     # (e.g. with pyparsing or using a third party library).
#     #
#     warnings.warn('Use of custom parameters parsing is deprecated, use OMPC', DeprecationWarning)
#     params = params_string
#     # Manage damn matlab triple quotes special case
#     params = matlab_triple_quotes_to_python(params)
#     # Group together {}, []; does not support nesting
#     groups = re.split(r'(\{.*?\}|\[.*?\]|".*")', params)
#     # Postprocess the groups
#     parameter_values = []
#     for group in groups:
#         if group.startswith('{') or group.startswith('['):
#             group = group.replace('{', '(').replace('}', ')')   # matlab cell arrays are not sets
#             parameter_values.append(matlab_val_to_python_val(group))
#         elif group.startswith('"'):  # Darn matlab triple quotes
#             parameter_values.append(matlab_val_to_python_val(group))
#         else:
#             for value in group.split(','):  # Let's assume no comma appear in matlab strings...
#                 if value.strip():
#                     parameter_values.append(matlab_val_to_python_val(value.strip()))
#     return parameter_values
#
#####################################################################
