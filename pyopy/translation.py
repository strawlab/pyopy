# coding=utf-8
"""Matlab source code parsing and manipulation.
These functions are very ad-hoc, but work for all our cases.
"""
from __future__ import print_function, absolute_import
import codecs
import re
import os.path as op
import numpy as np
from arpeggio import StrMatch, RegExMatch, ZeroOrMore, Optional, ParserPython, EOF, PTNodeVisitor, visit_parse_tree, \
    OneOrMore
from pyopy.base import MatlabSequence, MatlabId


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


def build_matlabcall_parser(reduce_tree=False, debug=False):
    """
    Parameters
    ----------
    reduce_tree : boolean, default False
      Check `arpeggio.ParserPython` documentation

    debug : boolean, default False
      Check `arpeggio.ParserPython` documentation (N.B. will generate dot files in the running directory)

    Returns
    -------
    The arpeggio parser. Call `parser.parse` to generate the AST.
    """

    def an_id():
        return RegExMatch(r'[_A-Za-z][_a-zA-Z0-9]*')

    def a_number():
        return [RegExMatch('-?\d+((\.\d*)?((e|E)(\+|-)?\d+)?)?'),
                StrMatch('-inf'), StrMatch('-Inf'), StrMatch('inf'), StrMatch('Inf'),
                StrMatch('nan'), StrMatch('NaN')]

    def a_string():
        return StrMatch("'"), RegExMatch("(''|[^'])*"), StrMatch("'")

    def a_sequence():
        return [
            (a_number, StrMatch(':'), a_number, StrMatch(':'), a_number),
            (a_number, StrMatch(':'), a_number)
        ]

    def a_comma_or_space():
        return [StrMatch(','), StrMatch('')]

    def an_empty_matrix():
        return StrMatch('['), StrMatch(']')

    def a_matrix_row():
        parts = [a_sequence, a_number, (StrMatch('['), a_matrix_row, StrMatch(']'))]
        return parts, ZeroOrMore(a_comma_or_space, parts)

    def a_2d_matrix():
        return StrMatch('['), a_matrix_row, OneOrMore(StrMatch(';'), a_matrix_row), StrMatch(']')

    def a_2d_matrices_concat():
        return StrMatch('['), a_2d_matrix, OneOrMore(a_comma_or_space, a_2d_matrix), StrMatch(']')

    def a_matrix():
        return [an_empty_matrix,
                (StrMatch('['), a_matrix_row, StrMatch(']')),
                a_2d_matrix,
                a_2d_matrices_concat]

    def an_empty_cell():
        return StrMatch('{'), StrMatch('}')

    def a_cell_row():
        return a_value, ZeroOrMore(a_comma_or_space, a_value)

    def a_2d_cell():
        return StrMatch('{'), a_cell_row, OneOrMore(StrMatch(';'), a_cell_row), StrMatch('}')

    def a_cell():
        return [an_empty_cell,
                (StrMatch('{'), a_cell_row, StrMatch('}')),
                a_2d_cell]

    def a_value():
        # N.B. do not change order
        return [a_sequence, a_number,
                a_signature_or_call, a_string, an_id,
                a_matrix, a_cell]

    def a_parameter_list():
        return a_value, ZeroOrMore(StrMatch(','), a_value)

    def a_signature_or_call():
        return an_id, StrMatch('('), Optional(a_parameter_list), StrMatch(')')

    def function_signature_call():
        return a_signature_or_call, EOF

    return ParserPython(function_signature_call, reduce_tree=reduce_tree, debug=debug)


class MatlabcallTreeVisitor(PTNodeVisitor):
    """
    A tree visitor for matlab calls that return a python representation of the call.

    Parameters
    ----------
    debug : boolean, default False
      Use debug output when visiting a tree

    Returns
    -------
    """

    def __init__(self, debug=False):
        # N.B. these actions assume that syntactic noise is being ignored, therefore defaults=True
        super(MatlabcallTreeVisitor, self).__init__(defaults=True, debug=debug)

    @staticmethod
    def visit_an_id(node, _):
        return MatlabId(node.value)

    @staticmethod
    def visit_a_string(_, children):
        if not children:
            return ''
        return children[0].replace("''", "'")

    @staticmethod
    def visit_a_number(node, _):
        try:
            return int(node.value)
        except ValueError:
            return float(node.value)

    @staticmethod
    def visit_a_sequence(_, children):
        return MatlabSequence(':'.join(map(str, children)))

    @staticmethod
    def visit_a_comma_or_space(*_):
        # Ignore syntax noise, when syntax noise includes spaces (as in matrix strings)
        return None

    @staticmethod
    def visit_an_empty_matrix(*_):
        return np.array([])

    @staticmethod
    def visit_a_matrix_row(_, children):
        return np.hstack(map(np.atleast_1d, children))

    @staticmethod
    def visit_a_2d_matrix(_, children):
        return np.vstack(children)

    @staticmethod
    def visit_a_2d_matrices_concat(_, children):
        return np.hstack(children)

    @staticmethod
    def visit_an_empty_cell(*_):
        return np.array([], dtype=np.object)

    @staticmethod
    def visit_a_cell_row(_, children):
        return np.array(list(children), dtype=np.object)

    @staticmethod
    def visit_a_2d_cell(_, children):
        return np.vstack(children)

    @staticmethod
    def visit_a_value(_, children):
        return children[0]

    @staticmethod
    def visit_a_parameter_list(_, children):
        return children

    @staticmethod
    def visit_a_signature_or_call(_, children):
        if len(children) == 1:
            return children[0], []
        return children[0], children[1]


def parse_matlab_function(function_or_call,
                          parser=build_matlabcall_parser(),
                          visitor=MatlabcallTreeVisitor()):
    # TODO: docstring this properly
    ast = parser.parse(function_or_call)
    return visit_parse_tree(ast, visitor)


if __name__ == '__main__':

    from pyopy.hctsa.hctsa_catalog import HCTSACatalog

    hctsa = HCTSACatalog()

    for opname, _ in hctsa.allops():
        operation = hctsa.operation(opname)

        # Concrete call
        matlab_call = operation.opcall
        try:
            name, params = parse_matlab_function(matlab_call)
        except:
            print(matlab_call)

        # This should work, but it does not...
        # matlab_function = matlab_function = operation.function.code
        # matlab_function = hctsa.functions_dict[operation.funcname].params
        # print(matlab_function)


# TODO: add @ matlab function handles
# TODO: we probably miss some escaping rules for matlab strings
