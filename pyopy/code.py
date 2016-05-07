# coding=utf-8
"""Matlab source code parsing and manipulation.
These functions are very ad-hoc, but work for all our cases.
"""
from __future__ import print_function, unicode_literals, absolute_import
import codecs
import re
import os.path as op


def matlab_funcname_from_filename(mfile):
    """Returns the name of the mfile.
    Matlab main functions must be named after the file they reside in.
    """
    return op.splitext(op.basename(mfile))[0]


def parse_matlab_funcdef(mfile,
                         funcdef_pattern=re.compile(r'^function\s*(\S.+)\s*=\s*(\S+)\s*\((.*)\)[\r\n]',
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
    a tuple prefunc (string), out (string), funcname (string), parameters (string list), postfunc (string)
    """
    expected_func_name = op.splitext(op.basename(mfile))[0]
    with codecs.open(mfile, encoding='utf-8') as reader:
        text = reader.read()  # .replace('\r\n', '\n')
        prefunc, out, funcname, parameters, postfunc = funcdef_pattern.split(text, maxsplit=1)
        if not funcname == expected_func_name:
            raise Exception('Problem parsing %s.\n'
                            'The function name does not correspond to the file name' %
                            mfile)
        parameters = [p.strip() for p in parameters.split(',')]
        return prefunc, out, funcname, parameters, postfunc


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
