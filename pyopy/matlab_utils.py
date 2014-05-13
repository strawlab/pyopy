# coding=utf-8
"""Ad-hoc parsing of m-files, matlab pimping and other goodies to deal with matlab/octave libraries."""
import os.path as op
import re
import numpy as np


###################################
# Manipulate matlab source code
###################################
# These are very ad-hoc, but work for all our cases.
# In the future we might want to go for a full-blown
# python matlab parser (like OMPC, libermate...)
###################################

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


def _float_or_int(val_string):
    """Given a string we know it is a number, get the float or int it represents."""
    if '.' in val_string:
        return float(val_string)
    return int(val_string)


class MatlabSequence(object):
    """Represents a matab sequence in python, keeping matlab semantics when expanding to numpy arrays.

    Parameters
    ----------
    msequence: string
        A string representing a matlab sequence (e.g. "1:2:10" or "[1.1:2.2:8.9]")

    Examples
    --------
    >>> ms = MatlabSequence(' [1 :3]')
    >>> ms.lower
    1
    >>> ms.upper
    3
    >>> ms.step
    1

    >>> ms = MatlabSequence('1.1:0.5:3.')
    >>> ms.lower
    1.1
    >>> ms.upper
    3.0
    >>> ms.step
    0.5

    This is not foolproof, but it should fail when passed a wrong matlab slice:
    >>> MatlabSequence('[a:3]')
    Traceback (most recent call last):
     ...
    Exception: [a:3] is not a proper matlab slice

    >>> MatlabSequence('you fool')
    Traceback (most recent call last):
     ...
    Exception: you fool is not a proper matlab slice
    """

    __slots__ = ['msequence', 'lower', 'upper', 'step']

    def __init__(self, msequence):
        super(MatlabSequence, self).__init__()
        self.msequence = msequence
        try:
            parts = self.msequence.split(':')
            if len(parts) == 2:
                lower, upper = parts
                self.lower = _float_or_int(lower.replace('[', '').strip())
                self.upper = _float_or_int(upper.replace(']', '').strip())
                self.step = 1
            elif len(parts) == 3:
                lower, step, upper = parts
                self.lower = _float_or_int(lower.replace('[', '').strip())
                self.upper = _float_or_int(upper.replace(']', '').strip())
                self.step = _float_or_int(step.strip())
            else:
                raise Exception('%s is not a proper matlab slice' % msequence)
        except Exception:
            raise Exception('%s is not a proper matlab slice' % msequence)

    def as_array(self):
        """
        Examples:

        >>> np.allclose(MatlabSequence('1:5').as_array(), [1, 2, 3, 4, 5])
        True

        >>> np.allclose(MatlabSequence('1.1:5').as_array(), [1.1, 2.1, 3.1, 4.1,])
        True

        >>> np.allclose(MatlabSequence('1.1:5.1').as_array(), [1.1, 2.1, 3.1, 4.1, 5.1])
        True

        >>> np.allclose(MatlabSequence('1:0.5:5').as_array(), [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
        True

        Floats and ints might not be interchangeable in matlab world
        This function should keep the type:

        >>> np.issubdtype(MatlabSequence('1:5').as_array().dtype, int)
        True
        """
        # We are advised not to use arange with floats, but it works well here...
        array = np.arange(start=self.lower, stop=self.upper + self.step, step=self.step)
        if len(array) and array[-1] > self.upper:
            return array[:-1]
        return array

    def matlab_sequence_string(self):
        """Returns the original matla sequence string.

        Examples
        --------
        >>> MatlabSequence('1.1:5').matlab_sequence_string()
        '1.1:5'
        """
        if self.step != 1:
            return '%r:%r:%r' % (self.lower, self.step, self.upper)
        return '%r:%r' % (self.lower, self.upper)

    def __repr__(self):
        return 'MatlabSequence(\'%s\')' % self.matlab_sequence_string()


def matlab_val_to_python_val(val_string):
    """Given a string representing a matlab value, create an equivalent python value.

    Parameters
    ----------
    val_string: string
        A string representing a matlab value (e.g. "[1, 2, 3]", "1.1" or "1:3")

    Returns
    -------
    The value in python-land.

    Examples
    --------
    >>> matlab_val_to_python_val('1.1')
    1.1
    >>> matlab_val_to_python_val('[1, 2, 3]')
    [1, 2, 3]
    >>> matlab_val_to_python_val('[1:2:3]')
    MatlabSequence('1:2:3')
    """
    try:
        return MatlabSequence(val_string)
    except:
        return eval(val_string)  # FIXME whoooooo... code injections...


def parse_matlab_params(params_string):
    """Parses a matlab parameter values string and return each value in python land.

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
    >>> parse_matlab_params("0.5,'best'")
    [0.5, 'best']
    >>> parse_matlab_params("'ar','''R'',2,''M'',1,''P'',2,''Q'',1'")
    ['ar', "'R',2,'M',1,'P',2,'Q',1"]
    """
    #
    # A note on parsing hardness:
    #   - csv does not support adding different opening and closing quote characters
    #   - of course regexps per se are useless for balanced delimiters matching,
    # So if we detect nested structures we will need to go for a full parser
    # (e.g. with pyparsing or using a third party library).
    #
    params = params_string
    # Manage damn matlab triple quotes special case
    while "'''" in params:
        pre, _, post = params.partition("'''")
        post = '"\'' + post.\
            replace("''", '33matlabscapedquotesnastiness4$'). \
            replace("'", '"'). \
            replace('33matlabscapedquotesnastiness4$', "'")  # Usefully nasty
        params = pre + post
    # Group together {}, []; does not support nesting
    groups = re.split(r'(\{.*\}|\[.*\]|".*")', params)
    # Postprocess the groups
    parameter_values = []
    for group in groups:
        if group.startswith('{') or group.startswith('['):
            group = group.replace('{', '(').replace('}', ')')   # matlab cell arrays are not sets
            parameter_values.append(matlab_val_to_python_val(group))
        elif group.startswith('"'):  # Darn matlab triple quotes
            parameter_values.append(matlab_val_to_python_val(group))
        else:
            for value in group.split(','):  # Let's assume no comma appear in matlab strings...
                if value.strip():
                    parameter_values.append(matlab_val_to_python_val(value.strip()))
    return parameter_values


###################################
# Use matlab/octave as computational engines
###################################
# There are many options to pimp matlab/octave to do the computational work:
#   http://stackoverflow.com/questions/9845292/a-tool-to-convert-matlab-code-to-python
# At the moment all these solutions  are really slow (call and data transfer overhead)
# when compared with analogous "official" solutions for other languages
# (e.g. Matlab Builder / Matlab Builder Ja).
# Is this an inherent problem with CPython?
###################################


class MatlabVar(object):
    """Represents a variable already in (matlab|octave)-land."""
    def __init__(self, engine, varname):
        super(MatlabVar, self).__init__()
        self.engine = engine
        self.varname = varname

    def get(self):
        return self.engine.get(self.varname)


def py2matlabstr(pyvar):
    """Returns a string representation of the python variable pyvar suitable for a call in matlab.
    If pyvar is a MatlabVar, returns the name in matlab-land.
    """
    if isinstance(pyvar, MatlabVar):
        return pyvar.varname
    if isinstance(pyvar, MatlabSequence):
        return pyvar.matlab_sequence_string()
    return pyvar.__repr__()


class MatlabEngine(object):
    """Unify APIs of different python->(octave|matlab)->python integration tools.
    This API should allow also to:
      - Playing well with MatlabVar
      - Playing well with MatlabSequence
      - Playing well with Matlab weird triple-quoted escaped strings
    """

    def __init__(self, engine_location=None):
        super(MatlabEngine, self).__init__()
        self.session(engine_location)
        self.warmup()

    def warmup(self):
        self.run_text('ones(1);')

    def run_text(self, command):
        return self._run(command)

    def run_function(self, funcname, *args):
        # TODO: magically get the return arity, as oct2py does looking at the bytecode
        #       but we probably do not want that magic
        command = 'pyengout74383744=%s(%s);' % (funcname, ','.join(map(py2matlabstr, args)))
        self._run(command)
        return self.get('pyengout74383744')

    def _run(self, command, *args):
        raise NotImplementedError()

    def put(self, varname, value):
        raise NotImplementedError()

    def get(self, varname):
        raise NotImplementedError()

    def session(self, engine_location=None):
        """Returns the backend session, opening it if necessary."""
        raise NotImplementedError()

    def add_path(self, path, recursive=True, begin=False, force=False):
        """Adds a path to a matlab engine."""
        path = op.abspath(path)
        self.run_text('matlab_path=path()')  # Needed for engines that do not return the result (e.g. pymatlab)
                                             # Reimplement the simple way...
        matlab_path = self.get('matlab_path')
        if path not in matlab_path or force:
            if recursive:
                self.run_text('generated_path=genpath(\'%s\')' % path)
                path = self.get('generated_path')
            self.run_text('addpath(\'%s\', \'%s\')' % (path, '-begin' if begin else '-end'))


class Oct2PyEngine(MatlabEngine):

    def __init__(self, engine_location=None):
        self._session = None
        super(Oct2PyEngine, self).__init__(engine_location)

    def run_function(self, funcname, *args):
        return getattr(self.session(), funcname)(*args)

    def _run(self, command, *args):
        return self.session().run(command)

    def put(self, varname, value):
        self.session().put(varname, value)
        return MatlabVar(self, varname)

    def get(self, varname):
        return self.session().get(varname)

    def session(self, engine_location=None):
        if self._session is None:
            from pyopy.oct2py_utils import Oct2PyNotAll
            self._session = Oct2PyNotAll()
        return self._session


class PyMatlabEngine(MatlabEngine):

    # Until this is properly implemented some quick and dirty conclusions:
    #
    # Octave: I only tried oct2py. It has quite an overhead on calls.
    #   - oct2py is based on expect, I think there comes a lot of the performance penalty.
    #   - also it is based on files; in my tests I used a ramdisk (tmpfs).
    #     but still: open, write, close, open, write, close files per call can be too heavy with short calls.
    #
    # Matlab:
    #   - pymatlab showed best performance, on a par with oct2py (or better, in-memory transfers?)
    #     still initialization is cumbersome, shows splash screen and requires tcsh installed
    #     also it is failing a lot with some code
    #     if we stick with it will need to patch a bit its code
    #   - pymatbridge is promising, but I cannot yet grasp well its approach
    #       http://arokem.github.io/python-matlab-bridge/
    #       https://github.com/arokem/python-matlab-bridge
    #       This looks promising: https://github.com/arokem/python-matlab-bridge/pull/63
    #       But it is really slow transferring not so big matrices (it all boils down to json via zmq)...
    #   - good-old mlabwrap seems slower than pymatlab
    #     might just stick with it as it is the only one providing some success
    #   - mlabwrap purepy is slow and verbose on the matlab session
    #   - mlab was an awful experience, matlab process claiming stdin and really slow
    #

    def __init__(self):
        raise Exception('Buggy, look for alternatives')
        self._session = None
        super(PyMatlabEngine, self).__init__()

    def warmup(self):
        self.session().run('ones(1);')

    def _run(self, command, *args):
        return self.session().run(command)

    def put(self, varname, value):
        self.session().putvalue(varname, value)
        return MatlabVar(self, varname)

    def get(self, varname):
        return self.session().getvalue(varname)

    def session(self, engine_location=None):
        if self._session is None:
            import pymatlab
            self._session = pymatlab.session_factory()
        return self._session


def engines_benchmark():
    """Benchmarks the engines available in this machine."""
    raise NotImplementedError()


###################################
# Entry point
###################################

if __name__ == '__main__':

    def doctests():
        import doctest
        doctest.testmod()

    import argh
    parser = argh.ArghParser()
    parser.add_commands([doctests])
    parser.dispatch()