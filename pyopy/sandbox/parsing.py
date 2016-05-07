from __future__ import print_function
import numpy as np
from pyopy.base import MatlabSequence, MatlabId
from arpeggio import StrMatch, RegExMatch, ZeroOrMore, Optional, ParserPython, EOF, PTNodeVisitor, visit_parse_tree, \
    OneOrMore


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
