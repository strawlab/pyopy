from __future__ import print_function
import numpy as np
from pyopy.base import MatlabSequence
from arpeggio import StrMatch, RegExMatch, ZeroOrMore, Optional, ParserPython, EOF, PTNodeVisitor, visit_parse_tree


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
                StrMatch('nan'), StrMatch('NaN'),
                (StrMatch('['), a_number, StrMatch(']')),
                (StrMatch('{'), a_number, StrMatch('}'))]

    def a_string():
        return StrMatch("'"), RegExMatch("(''|[^'])*"), StrMatch("'")

    def a_sequence():
        return [
            (a_number, StrMatch(':'), a_number, StrMatch(':'), a_number),  # 0:0.05:0.95
            (a_number, StrMatch(':'), a_number)  # 1:3
        ]

    def a_matrix_row():
        # a bit hacky regexp due to:
        #   matlab allowing things like [1 2], so a blank is equivalent to a comma here
        #   we are configuring arpeggio to ignore blanks
        return [a_sequence, a_matrix, a_number], ZeroOrMore(RegExMatch(',|'), [a_sequence, a_matrix, a_number])

    def a_matrix_row_list():
        return a_matrix_row, ZeroOrMore(StrMatch(';'), a_matrix_row)

    def a_matrix():
        return StrMatch("["), Optional(a_matrix_row_list), StrMatch("]")

    def a_cell_row():
        # Same syntax as in a matrix row
        # Can any value be a cell element?
        return a_value, ZeroOrMore(RegExMatch(',|'), a_value)

    def a_cell_row_list():
        return a_cell_row, ZeroOrMore(StrMatch(';'), a_cell_row)

    def a_cell():
        return StrMatch('{'), Optional(a_cell_row_list), StrMatch('}')

    def a_value():
        return [a_signature_or_call, a_sequence, a_matrix, a_cell, a_number, a_string, an_id]

    def a_parameter_list():
        return a_value, ZeroOrMore(StrMatch(','), a_value)

    def a_signature_or_call():
        return an_id, StrMatch('('), Optional(a_parameter_list), StrMatch(')')

    def function_signature_call():
        return a_signature_or_call, EOF

    return ParserPython(function_signature_call, reduce_tree=reduce_tree, debug=debug)


class MatlabId(object):

    __slots__ = 'name'

    def __init__(self, name):
        super(MatlabId, self).__init__()
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.name)

    def __eq__(self, other):
        return isinstance(other, MatlabId) and self.name == other.name


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
        return children[0].replace("''", "'")

    @staticmethod
    def visit_a_number(node, _):
        try:
            return int(node.value)
        except ValueError:
            return float(node.value)

    @staticmethod
    def visit_a_sequence(_, children):
        if 2 <= len(children) <= 3:
            return MatlabSequence(':'.join(map(str, children)))
        else:
            raise ValueError('A matlab sequence/slice must have two or three children')

    @staticmethod
    def visit_a_matrix_row(_, children):
        def e2array(e):
            try:
                return e.as_array()
            except AttributeError:
                return e
        values = children[::2]  # [::2] to ignore commas / spaces
        return np.hstack(map(e2array, values))

    @staticmethod
    def visit_a_matrix_row_list(_, children):
        return np.vstack(children)

    @staticmethod
    def visit_a_matrix(_, children):
        if not children:
            return np.array([])
        return np.vstack(children)

    @staticmethod
    def visit_a_cell(_, children):
        return np.array(list(children[0]), dtype=np.object)

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
        print(matlab_call)
        name, params = parse_matlab_function(matlab_call)

        # This should work, but it does not...
        # matlab_function = matlab_function = operation.function.code
        # matlab_function = hctsa.functions_dict[operation.funcname].params
        # print(matlab_function)


# TODO: add @ matlab function handles
# TODO: Are inner functions / closures slower when creating a parser?
#       Most probably this is irrelevant, but a benchmark is required.
# TODO: We probably miss some escaping rules for matlab strings
