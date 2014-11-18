# coding=utf-8
"""Ad-hoc parsing of m-files, matlab pimping and other goodies to deal with matlab/octave libraries."""
import atexit
from functools import partial
import os.path as op
import re
import shutil
import tempfile
import os
import uuid
import sys

import numpy as np
from scipy.io import savemat
from whatami import whatable, is_iterable

from oct2py import Struct
from oct2py.matread import MatRead
from oct2py import Oct2PyError, Oct2Py
from oct2py.matwrite import MatWrite, putvals, putval
from oct2py.utils import create_file
from pyopy.externals.ompc.ompcply import translate_to_str


# --- Matlab source code manipulation
# These are very ad-hoc, but work for all our cases.


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
        """Returns the original matlab sequence string.

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

    def __eq__(self, other):
        return self.msequence == other.msequence


def parse_matlab_params(matlab_params_string):
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
    def ompc2evaluable(opmc_string, evaluate=True):

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
            opmc_string = flatten(opmc_string, flatter)
        if not evaluate:
            return opmc_string
        pyval = eval(opmc_string)   # literal_eval does not cover MatlabSequence
        # if not isinstance(pyval, list):
        #     return pyval
        try:
            return tuple(np.array(pyval, dtype=np.float))  # FIXME: document: all numbers as doubles, allow disabling
            # N.B. numeric scalars also finally get converted to double, but this is too clever
            # better make explicit
        except:
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


def py2matstr(pyvar):
    """Returns a string representation of the python variable pyvar suitable for a call in matlab.
    If pyvar is a EngineVar, returns the name in matlab-land.
    """
    if isinstance(pyvar, EngineVar):
        return pyvar.name
    if isinstance(pyvar, MatlabSequence):
        return pyvar.matlab_sequence_string()
    if isinstance(pyvar, bool):
        return '1' if pyvar else '0'
    return pyvar.__repr__()


def python_val_to_matlab_val(pythonval):
    """Synonym to py2matstr."""
    return py2matstr(pythonval)


def ints2floats(*args):
    """Returns a list with args where integers have been converted to floats.
    In python, '2' is an int; in matlab, it is a double, and that can give problems with matlab's picky type system.
    """
    return map(lambda x: float(x) if isinstance(x, int) else x, args)


# --- Stuff to generate identifiers

def strings_generator(prefix='', suffix=''):
    from string import digits, ascii_uppercase, ascii_lowercase
    from itertools import product

    chars = digits + ascii_uppercase + ascii_lowercase

    for n in xrange(1, 1000):
        for comb in product(chars, repeat=n):
            yield prefix + ''.join(comb) + suffix


def some_strings(n, as_list=False, prefix='', suffix=''):
    from itertools import islice
    if as_list:
        return list(islice(strings_generator(prefix=prefix, suffix=suffix), n))
    return islice(strings_generator(), n)


# --- Common API for matlab/octave as computational engines


class EngineResponse(object):
    """Represents the response given by a computational engine to a request."""

    #
    # Modeled after responses from pymatbridge which are, at the moment:
    # { 'content': {'datadir', 'code', 'figures', 'stdout'},
    #   'success': 'true'}
    #

    def __init__(self,
                 success=False,
                 figures=None,
                 code=None,
                 datadir=None,
                 stdout=None):
        super(EngineResponse, self).__init__()
        self.success = success.lower() != u'false' if isinstance(success, (unicode, basestring)) else success
        self.datadir = datadir
        self.code = code
        self.figures = figures
        self.stdout = stdout

    @classmethod
    def from_pymatbridge_response(cls, response):
        return cls(success=response[u'success'],
                   figures=response[u'content'].get(u'figures', None),
                   code=response[u'content'].get(u'code', None),
                   datadir=response[u'content'].get(u'datadir', None),
                   stdout=response[u'content'].get(u'stdout', None))


class EngineException(Exception):
    """A regular exception carrying an EngineResponse object."""
    def __init__(self, engine_response, cause, message_prefix=''):

        self.engine_response = engine_response
        self.cause = cause
        message = u'%sEngine failed to run: %s\n\tEngine Reason: %s\ncaused by %s' % \
                  (message_prefix, engine_response.code, engine_response.stdout, repr(cause))
        super(EngineException, self).__init__(message)


class EngineVar(object):
    """Represents a variable already in (matlab|octave)-land.

    The key reason why we bother with defining this class at all is to be able
    to differentiate variable names from strings in python land.
    """

    def __init__(self, engine, name):
        super(EngineVar, self).__init__()
        self.eng = engine
        self.name = name

    def get(self):
        return self.eng.get(self)

    def clear(self):
        self.eng.clear(self)

    def exists(self):
        return self.eng.exists(self)

    def matlab_class(self):
        return self.eng.matlab_class(self)


def _segregate_evs(varnames, values):
    not_ev_values = [value for value in values if not isinstance(value, EngineVar)]
    not_ev_names = [name for name, value in zip(varnames, values) if not isinstance(value, EngineVar)]
    ev_values = [value for value in values if isinstance(value, EngineVar)]
    ev_names = [name for name, value in zip(varnames, values) if isinstance(value, EngineVar)]
    return not_ev_names, not_ev_values, ev_names, ev_values


def _segregate_mss(varnames, values):
    not_ms_values = [value for value in values if not isinstance(value, MatlabSequence)]
    not_ms_names = [name for name, value in zip(varnames, values) if not isinstance(value, MatlabSequence)]
    ms_values = [value for value in values if isinstance(value, MatlabSequence)]
    ms_names = [name for name, value in zip(varnames, values) if isinstance(value, MatlabSequence)]
    return not_ms_names, not_ms_values, ms_names, ms_values


@whatable
class MatlabEngine(object):
    """Simple unifying API to wrap different python->(octave|matlab)->python integration tools.

    This API should allow also to:
      - Playing well with EngineVar
      - Playing well with MatlabSequence
      - Playing well with Matlab weird triple-quoted escaped strings
      - Keep track of existing variables and try to tame memory consumption on matlab land

    Not goals of this class:
      - it does not try to make particularly efficient small data transfers or call dispatch
        although it should compare ok with other solutions with more magic
      - it does not try to wrap matlab calls in a pythonic way, we use strings
        (use directly oct2py, pymatbridge and the like for that)
    """

    def __init__(self, engine_location=None, int2float=True, warmup=False):
        super(MatlabEngine, self).__init__()
        self.int2float = int2float
        self._matlab_vars = {}
        self._num_results = 0
        self._num_variable_contexts = 0
        self._engine_location = engine_location
        if warmup:
            self.session()
            self.warmup()

    def warmup(self):
        """Initializes the engine to avoid delays in posterior calls."""
        self.run_command('ones(1);')

    # --- Command running

    def run_command(self,
                    command,
                    outs2py=False,
                    supress_text_output=True):
        """Runs the command, optionally retrieving its results into python land.

        Parameters
        ----------
        command: string
          A command to run in the engine session (e.g. 'ones(1, x);')

        outs2py: boolean, default False
          If True, the result of the command is brought into python-land.

        supress_text_output: boolean, default True
          If True, a semicolon is appended to de command to supress text output from the matlab interpreter

        Returns
        -------
        A tuple (matlab_response, result)
        The result of running the command (ans or the output variables) as a list if outs2py is True.
        None otherwise

        Raises
        ------
        EngineException if there is a problem on the matlab side,
        a regular exception if the problem is on the python side.
        """
        response = None
        try:
            command = unicode(command)
            response, actual_command = self._run_command_hook(
                command + (';' if supress_text_output and not command.endswith(';') else ''))
            if response.success is False:
                raise Exception(response.stdout)
            if outs2py:
                # return response, map(self.get, outputs_from_command(actual_command))
                outputs = self.get(outputs_from_command(actual_command))
                return response, outputs if isinstance(outputs, (list, tuple)) else (outputs,)
            return response, None
        except Exception, e:
            if response is None:
                raise
            _, _, traceback = sys.exc_info()  # Missing py3 exception chaining
            raise EngineException, (response, e), traceback

    def _run_command_hook(self, command):
        """Hook method that is called by run_command, returns a tuple (MatlabResponse, run_command_string)."""
        raise NotImplementedError()

    # --- Function running

    def run_function_fast(self, nout, funcname, *args):
        """Same as run_function, but testing optimisations like:
          - do not move parameter values but instead encode them in function call (should be default when working)
          - do not release result_vars / allow keeping them and specifying their names
        """
        result_vars = ['pyopy_result_%d' % i for i in xrange(self._num_results, self._num_results + nout)]
        try:
            if not is_iterable(args):  # split apart numpy arrays, move these using a put context
                args = [args]
            command = u'%s=%s(%s);' % (u'[%s]' % u','.join(result_vars), funcname, u','.join(map(py2matstr, args)))
            _, results = self.run_command(command, outs2py=True)
            return results[0] if len(results) == 1 else results
        finally:
            self.clear(result_vars)  # we could also avoid this burden too

    def run_function(self, nout, funcname, *args):
        """Runs a function and returns the first nout values.

        Parameters
        ----------
        nout: int
          The number of output arguments

        funcname: string
          The name of the function to execute

        args: values
          The arguments to the call. Everything will be moved to python land except for EngineVar values

        Returns
        -------
        Result(s) are the outputs already in python land (a single value if nout=1, else a list of values)

        Raises
        ------
        Same as run_command.
        """
        result_vars = ['pyopy_result_%d' % i for i in xrange(self._num_results, self._num_results + nout)]
        try:
            with self.context(args) as args:
                # N.B. we can make this more efficient easily by encoding parameters as literals in the call
                if not is_iterable(args):
                    args = [args]
                command = u'%s=%s(%s);' % (u'[%s]' % u','.join(result_vars), funcname, u','.join(map(py2matstr, args)))
                _, results = self.run_command(command, outs2py=True)
                return results[0] if len(results) == 1 else results
        finally:
            self.clear(result_vars)

    # --- Python -> Matlab

    def put(self, varnames, values, strategy=None):
        """Copies python values into matlab-land.

        Parameters
        ----------
        varnames: string or list of strings
          the name(s) of the variables in matlab land

        values: python value or list of python values
          the values of the parameters in python land

        strategy: string, default 'by_file'
          this might or might not modify how concrete engines transfer data

        Returns
        -------
        A EngineVar variable list linked to varnames with the proper values in matlab land.
        """
        # we require lists or tuples, not ducks...
        if not isinstance(varnames, (list, tuple)):
            varnames = [varnames]
        if not isinstance(values, (list, tuple)):
            values = [values]
        # matlab variables cannot start by _
        for name in varnames:
            if name.startswith('_'):
                raise Exception('Invalid name {0}'.format(name))
        # unicode
        varnames = map(unicode, varnames)
        # int -> float (beware of too much magic)
        if self.int2float:
            values = ints2floats(*values)
        # treat differently enginevar from other values
        not_ev_names, not_ev_values, ev_names, ev_values = _segregate_evs(varnames, values)
        not_ev_names, not_ev_values, ms_names, ms_values = _segregate_mss(not_ev_names, not_ev_values)
        # values not in matlab land get transfered...
        if len(not_ev_names) > 0:
            self._put_hook(not_ev_names, not_ev_values, strategy=strategy)
        # ...as opposed to values in matlab land, that get aliased (but careful with WOC)
        # and matlab-sequences, that get transferred as string
        if len(ev_names) > 0 or len(ms_names) > 0:
            command = ';'.join('%s=%s' % (alias, py2matstr(val)) for alias, val
                               in zip(ev_names, ev_values) + zip(ms_names, ms_values))
            self.run_command(command, outs2py=False)
        # return a single var...
        if len(varnames) == 1:
            return EngineVar(self, varnames[0])
        # ...or a list of vars
        return [EngineVar(self, varname) for varname in varnames]

    def _put_hook(self, varnames, values, strategy='by_file'):
        raise NotImplementedError()

    def get(self, varnames, strategy='by_file'):
        """Gets variables from matlab-land into python-land.

        Parameters
        ----------
        varnames: string or list of strings
          the name(s) of the variable(s) in matlab land


        strategy: string, default 'by_file'
          this might or might not modify how concrete engines transfer data

        Returns
        -------
        The variable values in python land (None for variables that does not exist in the matlab/octave session).
        """
        raise NotImplementedError()

    def session(self):
        """Returns the backend session, opening it if necessary."""
        raise NotImplementedError()

    def close_session(self):
        """Closes the backend session, if it is opened, shuttind down the engine."""
        raise NotImplementedError()

    def clear(self, varnames):
        """Cleans variables (string or EngineVariable instances) from the matlab/octave workspace."""
        if isinstance(varnames, (unicode, basestring, EngineVar)):
            varnames = [varnames]
        varnames = [var.name if isinstance(var, EngineVar) else var for var in varnames]
        self.run_command('clear %s' % ' '.join(varnames), outs2py=False)

    def mwho(self, in_global=False):
        """Returns a list with the variable names in the current matlab/octave workspace."""
        # Remember also whos, class, clear, clearvars...
        if in_global:
            return self.run_command('ans=who(\'global\');', outs2py=True)[1][0]  # run with ()
        return self.run_command('ans=who();', outs2py=True)[1][0]  # run with ()

    def list_variables(self, in_global=False):
        """mwho() synonym."""
        return self.mwho(in_global)

    def matlab_class(self, name):
        """Returns the class of a variable in matlab-land."""
        name = name.name if isinstance(name, EngineVar) else name
        return self.run_command('class(%s)' % name, outs2py=True)[1][0]

    def exists(self, name):
        """Returns True iff the variable exists in the engine session."""
        name = name.name if isinstance(name, EngineVar) else name
        return self.run_function(1, 'exist', name) == 1

    def add_path(self, path, recursive=True, begin=False, force=False):
        """Adds a path to this matlab engine.

        Parameters
        ----------
        path: string
          The directory we will add to the python session

        recursive: boolean, default True
          If True, adds subdirectories recursively

        begin: boolean, default False
          If True, add the path(s) to the beginning of the session system paths

        force: boolean, default False
          If True, the path(s) will be added even if they are already there;
          If False and "path" is already in the system directory, add nothing

        Returns
        -------
        The matlab path after this operation has completed.
        """
        path = op.abspath(path)
        _, (matlab_path,) = self.run_command('ans=path();', outs2py=True)
        if path not in matlab_path or force:
            if recursive:
                path = self.run_command('generated_path=genpath(\'%s\')' % path, outs2py=True)[1][0]
            self.run_command('addpath(\'%s\', \'%s\')' % (path, '-begin' if begin else '-end'))
        return self.run_command('ans=path();', outs2py=True)[1][0]

    # --- Context manager stuff

    def __enter__(self):
        self.session()
        return self

    def __exit__(self, etype, value, traceback):
        try:
            self.close_session()
        except:
            pass

    # --- Arbitrary cleanup stuff
    # FIXME: decide whether we want __del__ or not

    #
    # def __del__(self):
    #     self.__exit__(None, None, None)
    #
    #######

    # ---Engine variables contexts
    def context(self, values, varnames=None, alias_engine_vars=False):
        class EngineVariablesContext(object):
            def __init__(self, engine, context_id, values, varnames=None):
                super(EngineVariablesContext, self).__init__()
                self.engine = engine
                self.context_id = context_id
                if varnames is None:
                    varnames = []
                # n.b. EngineVar get an alias in matlab land
                # making sure we do not overwrite other vars is hard, so we just hardcode some names...
                if len(varnames) < len(values):
                    varnames += some_strings(len(values) - len(varnames),
                                             as_list=True,
                                             prefix='pyopy_context_%s_' % str(context_id))
                self.names = varnames
                self.values = values
                self.alias_engine_vars = alias_engine_vars

            def __enter__(self):
                if not self.alias_engine_vars:
                    not_ev_names, not_ev_values, ev_names, ev_values = \
                        _segregate_evs(self.names, self.values)
                    # Make aliased vars...
                    # push data to matlab land
                    evars = self.engine.put(not_ev_names, not_ev_values)
                    # merge
                    if isinstance(evars, list):
                        evars += ev_values
                    else:
                        evars = [evars] + ev_values
                    # reconstruct original order
                    order = [name if not isinstance(value, EngineVar) else value.name
                             for name, value in zip(self.names, self.values)]
                    sort_dict = {var.name: var for var in evars}
                    return [sort_dict[varname] for varname in order]
                return self.engine.put(self.names, self.values)

            def __exit__(self, etype, value, traceback):
                self.engine.clear(self.names)

        self._num_variable_contexts += 1  # useless at the moment
        return EngineVariablesContext(self, 0, values, varnames=varnames)
        #
        # N.B. hopefully we won't get in trouble by reusing variable names
        # (anyway octave is single-threaded and multiple python processes would use different octave instances)
        #
        # But in matlab we can get in trouble by creating too many different variable names (even if we clean them)
        # See: http://stackoverflow.com/questions/20328159/matlab-variable-count-limit
        #      http://www.mathworks.com/matlabcentral/newsreader/view_thread/241303
        #      https://www.mathworks.com/matlabcentral/newsreader/view_thread/158179
        #


#####################
#
# --- By default we will use the Oct2Py mechanism to transfer data
#
#####################
#
# transplant has a seemingly json-based fast protocol, might be interesting to see how it compares
# time allowing, explore other options
# best stuff typewise is, though, oct2py (it even supports sparse matrices) or hdf5storage
# at the moment hdf5storage is just slow...
#
#####################


# --- Hack to allow data already in (octave/matlab)-land to be reused

class MatWriteNotAll(MatWrite):

    # This is some copy&paste from oct2py version...
    __OCt2PY_COPY_PASTE_VER__ = '2.3.0'

    def create_file(self, inputs, names=None):
        """
        Create a MAT file, loading the input variables.

        If names are given, use those, otherwise use dummies.

        Parameters
        ==========
        inputs : array-like
            List of variables to write to a file.
        names : array-like
            Optional list of names to assign to the variables.

        Returns
        =======
        argin_list : str or array
            Name or list of variable names to be sent.
        load_line : str
            Octave "load" command.

        """
        # create a dummy list of var names ("A", "B", "C" ...)
        # use ascii char codes so we can increment
        argin_list = []
        ascii_code = 65
        data = {}
        for var in inputs:
            if isinstance(var, EngineVar):  # PYOPY: one change
                argin_list.append(var.name)
                continue
            if isinstance(var, MatlabSequence):  # PYOPY: another change
                argin_list.append(var.matlab_sequence_string())
                continue
            if names:
                argin_list.append(names.pop(0))
            else:
                argin_list.append("%s__" % chr(ascii_code))
            # for structs - recursively add the elements
            try:
                if isinstance(var, dict):
                    data[argin_list[-1]] = putvals(var)
                else:
                    data[argin_list[-1]] = putval(var)
            except Oct2PyError:
                raise
            ascii_code += 1
        if not os.path.exists(self.in_file):
            self.in_file = create_file(self.temp_dir)
        try:
            savemat(self.in_file, data, appendmat=False,
                    oned_as=self.oned_as, long_field_names=True)
        except KeyError:  # pragma: no cover
            raise Exception('could not save mat file')
        load_line = 'load {0} "{1}"'.format(self.in_file,
                                            '" "'.join(argin_list))
        return argin_list, load_line


class Oct2PyNotAll(Oct2Py):

    def __init__(self, executable=None, logger=None, timeout=None, oned_as='row', temp_dir=None):
        self._writer = None
        super(Oct2PyNotAll, self).__init__(executable, logger, timeout, oned_as, temp_dir)

    def restart(self):
        super(Oct2PyNotAll, self).restart()
        self._writer = MatWriteNotAll()  # Fixme: lame, ask for pull request of EngineVar + MatWrite


# --- Base class

class Oct2PyTransferEngine(MatlabEngine):

    def __init__(self,
                 engine_location=None,
                 tmp_dir_root=None,
                 tmp_prefix='pymatbridge_engine_',
                 warmup=False):
        # file based data transit
        self._tmpdir_root = tmp_dir_root
        self._tmp_prefix = tmp_prefix
        self._tmpdir = None
        self._matwrite = None
        self._matread = None
        super(Oct2PyTransferEngine, self).__init__(engine_location, warmup=warmup)

    # --- Matlab -> Python

    def get(self, varnames, strategy='oct2py'):
        # we require lists or tuples, not ducks...
        if not isinstance(varnames, (list, tuple)):
            varnames = [varnames]
        # for convenience we allow to pass EngineVars
        varnames = [name.name if isinstance(name, EngineVar) else name for name in varnames]
        # this is the easy, slow way
        if strategy is not None and strategy != 'oct2py':
            raise Exception('reading strategy %r for %s not supported, please use "oct2py" only' % (
                strategy, self.__class__.__name__))
        # Adapted from oct2py.pull (N.B. as side effect, varname gets empty, so copy)
        argout_list, save_line = self.matread().setup(len(varnames), list(varnames))
        self.run_command(save_line)
        data = self.matread().extract_file(variables=varnames)
        self.matread().remove_file()
        self._matread = None  # not the fastest, but avoids any leak
        if isinstance(data, dict) and not isinstance(data, Struct):
            return [data.get(v, None) for v in argout_list]
        else:
            return data

    # --- Python -> Matlab

    def _put_hook(self, varnames, values, strategy='oct2py'):
        # only oct2py stuff atm...
        if strategy is not None and not strategy == 'oct2py':
            raise Exception('writing strategy %r for %s not supported, please use "oct2py" only' % (
                strategy, self.__class__.__name__))
        _, load_line = self.matwrite().create_file(values, varnames)
        # matlab does not understand if it is quoted with "
        load_line = load_line.replace('"', "'")
        self.run_command(load_line)
        self.matwrite().remove_file()

    # --- Temporary files management

    def tempdir(self):
        if self._tmpdir is None:
            self._tmpdir = tempfile.mkdtemp(prefix=self._tmp_prefix, dir=self._tmpdir_root)
        return self._tmpdir

    def matwrite(self):
        if self._matwrite is None:
            self._matwrite = MatWriteNotAll(temp_dir=self.tempdir())
        return self._matwrite

    def matread(self):
        if self._matread is None:
            self._matread = MatRead(temp_dir=self.tempdir())
        return self._matread

    # --- Context manager

    def __exit__(self, etype, value, traceback):
        super(Oct2PyTransferEngine, self).__exit__(etype, value, traceback)
        if self._tmpdir is not None and op.isdir(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)


# --- Octave Oct2Py adaptor

class Oct2PyEngine(Oct2PyTransferEngine):

    def __init__(self,
                 engine_location='octave',
                 verbose=False,
                 timeout=None,
                 log=True,
                 tmp_dir_root=None,
                 warmup=False):
        self._session = None
        self._verbose = verbose
        self._timeout = timeout
        self._log = log
        super(Oct2PyEngine, self).__init__(engine_location=engine_location,
                                           tmp_dir_root=tmp_dir_root,
                                           tmp_prefix='oct2py_engine_',
                                           warmup=warmup)

    # --- Command running

    def _run_command_hook(self, command):

        ##########
        #
        # At the moment, docs for oct2py are not correct; two things can be returned from eval:
        #   - the response can be either text or ans
        #   - the value of "ans" in python-land (e.g. a numpy array if we run "ones(3)")
        #
        # So we need a dirty workaround to avoid "ans" to be moved to python land each time, if not requested
        #
        # TODO: report the inconsistency of outputs, wrong doc and
        #       not being able to get ans (which is an octave peculiarity)
        #
        ##########

        # Brittle, can fail in many ways..
        new_command = ('ans_pyopy = %s' % command) if outputs_from_command(command) == ['ans'] else command

        text_response = self.session().eval(new_command,
                                            verbose=self._verbose, timeout=self._timeout, log=self._log)

        return EngineResponse(success=True, code=command, stdout=text_response), new_command

    # --- Python -> Matlab

    def _put_hook(self, varnames, values, strategy=None):
        self.session().push(varnames, values, verbose=self._verbose, timeout=self._timeout)
        # N.B. we could unify

    # --- Session management

    def session(self):
        if self._session is None:
            self._session = Oct2PyNotAll(executable=self._engine_location)
        return self._session

    def close_session(self):
        try:
            self._session.close()
        finally:
            self._session = None


# --- PyMatBridge adaptor

class PyMatBridgeEngine(Oct2PyTransferEngine):

    def __init__(self,
                 engine_location=None,
                 tmp_dir_root=None,
                 sid=None,
                 octave=False,
                 warmup=False,
                 single_threaded=True):
        # session control
        self._session = None
        self._id = 'pymatbridge-' + str(uuid.uuid1()) if sid is None else sid
        self._octave = octave
        self._single_threaded = single_threaded
        if engine_location is None:
            engine_location = 'octave' if octave else 'matlab'
        # we transfer data using matfiles and oct2py conversion rules
        super(PyMatBridgeEngine, self).__init__(engine_location=engine_location,
                                                tmp_dir_root=tmp_dir_root,
                                                warmup=warmup)

    # --- Command running

    def _run_command_hook(self, command):
        return EngineResponse.from_pymatbridge_response(self.session().run_code(command)), command

    # --- Session management

    def session(self):
        if self._session is None:
            import pymatbridge
            # N.B. capture_stdout=True in John's branch
            # N.B. Octave requires version > 0.3 (git when writing this)
            eng = pymatbridge.Octave if self._octave else pymatbridge.Matlab
            if self._octave:
                startup_ops = None
            else:
                startup_ops = ' -nodesktop -nodisplay' if not self._single_threaded else \
                    ' -nodesktop -nodisplay -singleCompThread'
            self._session = eng(executable=self._engine_location,
                                id=self._id,
                                socket_addr='ipc:///tmp/%s' % self._id,
                                startup_options=startup_ops)
            self._session.start()
        return self._session

    def close_session(self):
        try:
            self._session.stop()
        finally:
            self._session = None

    # TODO: if ever needed for performance,refine put/get values, if scalar use json via zmq, if not use files


# --- Global engines

class Engines(object):

    _engines = {}
    _default = None

    @staticmethod
    def engine(name, thunk=None):
        if name not in Engines._engines:
            Engines._engines[name] = thunk()
        return Engines._engines[name]

    @staticmethod
    def matlab():
        return Engines.engine('matlab', thunk=PyMatBridgeEngine)

    @staticmethod
    def octave():
        return Engines.engine('octave', thunk=Oct2PyEngine)

    @staticmethod
    def close(name):
        if name in Engines._engines:
            try:
                Engines._engines[name].close_session()
            except:
                pass
            finally:
                del Engines._engines[name]

    @staticmethod
    def close_all():
        for _, v in Engines._engines.iteritems():
            try:
                v.close_session()
            except:
                pass
        Engines._engines = {}

    @staticmethod
    def close_matlab():
        Engines.close('matlab')

    @staticmethod
    def close_octave():
        Engines.close('octave')

    @staticmethod
    def default():
        try:
            print 'Using matlab...'
            return Engines.matlab()
        except:
            print 'Using octave...'
            return Engines.octave()

    @staticmethod
    def set_default(name, thunk):
        raise NotImplementedError('To implement ASAP')

atexit.register(Engines.close_all)

# --- Entry point

if __name__ == '__main__':

    def doctests():
        import doctest
        doctest.testmod()

    import argh
    parser = argh.ArghParser()
    parser.add_commands([doctests])
    parser.dispatch()


##############################
#
# Full blown python matlab parsers and translators:
#   - OMPC (we use it here for robust parsing of matlab statements)
#   - libermate
#   - Mat2py
#
####################
# Python <-> Octave/Matlab bridges notes
####################
#
# There are many options to pimp matlab/octave to do the computational work:
#   http://stackoverflow.com/questions/9845292/a-tool-to-convert-matlab-code-to-python
# At the moment all these solutions  are really slow (call and data transfer overhead)
# when compared with analogous "official" solutions for other languages
# (e.g. Matlab Builder / Matlab Builder Ja).
# Is this an inherent problem with CPython?
#
# Octave:
#   - oct2py:
#     - It has quite an overhead on calls. oct2py is based on expect,
#       where some of the performance penalty comes from
#     - also it has a lot of reflection and disassembling magic, which surely makes each call costly
#       probably this is the biggest bottleneck for programs based on many calls
#       this complexity could be tamed by providing explicitly the number of desired outputs
#       and then disabling introspection + dissassembling magic in this kind of calls
#     - also it is based on files; it is quite efficient, specially if using ramdisk (in my case, tmpfs)
#       and if compared with other serialisation stuff based on jsonish serializers.
#       Because no inmemory tunnel is ever used, no special connection with octave is required,
#       which makes the library pure-python (neat) and much easier to use.
#       For our use cases, file roundtrip works fast enough.
#     - I wonder how much cruft is left on the octave side per call
#     - has great roundtrip abilities, shared with its brother project sci2py
#
# Matlab:
#   - transplant is very new. It claims fast, only py3 at the moment.
#     https://github.com/bastibe/transplant
#   - pymatlab showed best performance, on a par with oct2py (or better, in-memory transfers?)
#     still initialization is cumbersome, shows splash screen and requires tcsh installed
#     also it is failing a lot with some code
#     if we stick with it will need to patch a bit its code
#   - pymatbridge is promising:
#       https://github.com/arokem/python-matlab-bridge
#     See also John's branches for num_out and auto-use-mat features:
#       https://github.com/nzjrs/python-matlab-bridge
#     Based on the promising:
#       https://github.com/arokem/python-matlab-bridge/pull/63
#     Caveat: it is really slow transferring not so big matrices (it all boils down to json via zmq)...
#             could be circunvented using files (ala Oct2py/John/hdfstorage...)
#             or maybe a better encoding (ala transplant)
#   - good-old mlabwrap seems slower than pymatlab
#   - mlabwrap purepy is slow and verbose on the matlab session
#   - mlab was an awful experience, matlab process claiming stdin and really slow
#   - manual command line (should prove easy)
#     matlab -nojvm -nodisplay -r "commands;quit"
#
####################
# Random Notes
####################
#
#   Probably it is better to apply many functions to same series first
#   (enhance cache locality), cellfun...
#
#   try...catch works in both octave and matlab
#
#   memmaps are only supported in matlab
#
#   This seems to have a bit more functionality than scipy savemat / loadmat and it is great for data roundtrip:
#       https://pypi.python.org/pypi/hdf5storage/0.1.3
#
####################
# TODOs
####################
#
# TODO: magically get the return arity, as oct2py does by looking at the bytecode (originally from OMPC)
#       but we probably do not want that magic, it is hugely slow
#       and at the moment, all funcs explored have arity 1 (from HCTSA)
#
# TODO: how much running things on pycharm affect performance?
#
# TODO: better control timeouts
#    it seems it is not simple in single-threaded matlab-land, if possible at all
#    we could go for python (signal or multiprocessing + kill matlab session)
#       http://eli.thegreenplace.net/2011/08/22/how-not-to-set-a-timeout-on-a-computation-in-python
#       http://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish
#       http://stackoverflow.com/questions/492519/timeout-on-a-python-function-call
#    oct2py already provides a session-wide timeout
#       http://blink1073.github.io/oct2py/source/info.html#timeout
#
####################
#
# this is a probably a valid get() implementation for pymatbridge
# it has semantic differences between oct2py and pymatbridge
# (e.g. last time I tested matrices where passed as pure python lists, instead of ndarrays)
# if len(varnames) == 1:
#     return self._session.get_variable(varnames[0])
# return map(self._session.get_variable, varnames)
#
####################
#
# More oct2pyish implementation of run_function and get
#
# def run_function(self, funcname, *args):
#     return getattr(self.session(), funcname)(*args)
#
# def get(self, varnames, strategy=None):
#     # we require lists or tuples, not ducks...
#     if not isinstance(varnames, (list, tuple)):
#         varnames = [varnames]
#     # get name from EngineVar objects
#     varnames = [var.name if isinstance(var, EngineVar) else var for var in varnames]
#     # oct2py does not like much non-unicode strings
#     varnames = map(unicode, varnames)
#     # transfer memory from matlab to python land
#     return self.session().pull(list(varnames))
#     # N.B. copy to workaround side effects of oct2py push (that pops from the original list!)
#
####################
#
# To compile the pymatbridge octave messenger:
#   mkoctfile --mex -lzmq messenger.c
# Put somewhere in the octave path
#
#
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