# coding=utf-8
"""Common API for matlab/octave as computational engines."""
import atexit
import copy
from itertools import izip
import os.path as op
import sys

import numpy as np
from pyopy.code import outputs_from_command

from pyopy.misc import ints2floats, some_strings, is_iterable
from pyopy.misc import float_or_int


# --- Engine variables: transparently reuse data in matlab land

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

    def engine_class(self):
        return self.eng.engine_class(self)


def _segregate_evs(varnames, values):
    not_ev_values = [value for value in values if not isinstance(value, EngineVar)]
    not_ev_names = [name for name, value in zip(varnames, values) if not isinstance(value, EngineVar)]
    ev_values = [value for value in values if isinstance(value, EngineVar)]
    ev_names = [name for name, value in zip(varnames, values) if isinstance(value, EngineVar)]
    return not_ev_names, not_ev_values, ev_names, ev_values


# --- Adapt matlab sequences syntax

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

    # Cope with the duality matlab/python efficiently (see py2matstr)
    USE_MATLAB_REPR = False

    def __init__(self, msequence):
        super(MatlabSequence, self).__init__()
        self.msequence = msequence
        try:
            parts = self.msequence.split(':')
            if len(parts) == 2:
                lower, upper = parts
                self.lower = float_or_int(lower.replace('[', '').strip())
                self.upper = float_or_int(upper.replace(']', '').strip())
                self.step = 1
            elif len(parts) == 3:
                lower, step, upper = parts
                self.lower = float_or_int(lower.replace('[', '').strip())
                self.upper = float_or_int(upper.replace(']', '').strip())
                self.step = float_or_int(step.strip())
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

    def python_sequence_string(self):
        return 'MatlabSequence(\'%s\')' % self.matlab_sequence_string()

    def copy_for_matlab(self):
        """Copes with the duality python/matlab on __repr__."""
        # ...this is is necessary for correct dispatch given our current implementation
        myself = copy.copy(self)
        myself.__repr__ = myself.matlab_sequence_string
        return myself

    def copy_for_python(self):
        """Copes with the duality python/matlab on __repr__."""
        # ...this is is necessary for correct dispatch given our current implementation
        myself = copy.copy(self)
        myself.__repr__ = myself.python_sequence_string
        return myself

    def __repr__(self):
        if self.USE_MATLAB_REPR:
            return self.matlab_sequence_string()
        return self.python_sequence_string()

    def __eq__(self, other):
        return self.msequence == other.msequence


#
# in octave, all these get flattened:
#   1:4         = [1, 2, 3, 4]
#   [1:4]       = [1, 2, 3, 4]
#   [1, 2:3, 4] = [1, 2, 3, 4]
#   [1:2, 3:4]  = [1, 2, 3, 4]
#
# However, [1:3] could get translated to something like
#    mcat([mslice[1:3]]) -> [MatlabSequence('1:3')]
#
# This is not so easy to deal with using our clumsy way of parsing
# So we will do in python land now, in an over-conservative way
#
# TODO: deal with nested MatlabSequences, could be done in MatlabWrite too
#       no use case at the moment, but it is not difficult to imagine...
#

def _flatten_matlab_sequences(pyval):
    if not isinstance(pyval, list) or not any(isinstance(x, MatlabSequence) for x in pyval):
        return pyval
    # unflatten MatlabSequences
    return '[%s]' % ','.join(x.matlab_sequence_string() if isinstance(x, MatlabSequence) else str(x) for x in pyval)


def _has_matlab_sequences(pyval):
    return isinstance(pyval, MatlabSequence) or \
        isinstance(pyval, list) and any(isinstance(x, MatlabSequence) for x in pyval)


def _segregate_mss(varnames, values):
    has_matlab = map(_has_matlab_sequences, values)
    not_ms_values = [value for value, hms in izip(values, has_matlab) if not hms]
    not_ms_names = [name for name, hms in izip(varnames, has_matlab) if not hms]
    ms_values = [value for value, hms in izip(values, has_matlab) if hms]
    ms_names = [name for name, hms in izip(varnames, has_matlab) if hms]
    return not_ms_names, not_ms_values, ms_names, ms_values


# --- Python variable represented as a matlab string

def py2matstr(pyval):
    """Returns a string representation of the python value suitable for a call in matlab.
    If pyval is a EngineVar, returns the name in matlab-land.
    Things like [MatlabSequence('1:3'), 2] should also get proper matlab representations ('[1:3, 2]')
    """
    old_msequence_config = MatlabSequence.USE_MATLAB_REPR  # mmm global state, not thread safe... fix if ever a problem
    try:
        MatlabSequence.USE_MATLAB_REPR = True
        if isinstance(pyval, EngineVar):
            return pyval.name
        if isinstance(pyval, bool):
            return '1' if pyval else '0'
        return pyval.__repr__()
    finally:
        MatlabSequence.USE_MATLAB_REPR = old_msequence_config


def python_val_to_matlab_val(pythonval):
    """Synonym to py2matstr."""
    return py2matstr(pythonval)


# --- Responses from matlab land

class EngineResponse(object):
    """Represents the response given by a computational engine to a request."""

    def __init__(self,
                 success,
                 code=None,
                 stdout=None,
                 stderr=None,
                 exception=None,
                 **kwargs):
        super(EngineResponse, self).__init__()
        self.code = code
        self.success = success.lower() != u'false' if isinstance(success, (unicode, basestring)) else success
        self.stdout = stdout
        self.stderr = stderr
        self.exception = exception
        self.extra = kwargs

    @classmethod
    def from_pymatbridge_response(cls, response):
        #
        # Modeled after responses from pymatbridge which are, at the moment:
        # { 'content': {'datadir', 'code', 'figures', 'stdout'},
        #   'success': 'true'}
        #
        return cls(success=response[u'success'],
                   figures=response[u'content'].get(u'figures', None),
                   code=response[u'content'].get(u'code', None),
                   datadir=response[u'content'].get(u'datadir', None),
                   stdout=response[u'content'].get(u'stdout', None))

    def matlab_out(self):
        if self.stdout is None and self.stderr is None:
            return None
        return '# --- stdout:\n%s\n\n# --- stderr:\n%s' % (
            self.stdout if self.stdout is not None else '',
            self.stderr if self.stderr is not None else '',
        )


# --- Exceptions caused by the engine

class EngineException(Exception):
    """A regular exception carrying an EngineResponse object."""
    def __init__(self, engine_response, cause, message_prefix=''):
        self.engine_response = engine_response
        self.cause = cause
        message = u'%sEngine failed to run: %s\n\tEngine Reason: %s\ncaused by %s' % \
                  (message_prefix, engine_response.code, engine_response.matlab_out(), repr(cause))
        super(EngineException, self).__init__(message)


# --- Data transfer between python and matlab


class PyopyTransplanter(object):

    def __init__(self, engine=None, int2float=True):
        super(PyopyTransplanter, self).__init__()
        self._engine = engine
        self._int2float = int2float

    def get(self, varnames, eng):
        """Gets variables from matlab-land into python-land.

        Parameters
        ----------
        varnames: string or list of strings
          the name(s) of the variable(s) in matlab land

        eng: A MatlabEngine instance of None
          if None, the default engine (set via the constructor) will be used

        Returns
        -------
        The variable values in python land (None for variables that does not exist in the matlab/octave session).
        """
        eng = eng if eng is not None else self._engine
        # we require lists or tuples, not ducks...
        if not isinstance(varnames, (list, tuple)):
            varnames = [varnames]
        # for convenience we allow to pass EngineVars
        varnames = [name.name if isinstance(name, EngineVar) else name for name in varnames]
        # interact with the engine and return
        return self._get_hook(varnames, eng)

    def _get_hook(self, varnames, eng):
        raise NotImplementedError()

    def put(self, varnames, values, eng, int2float=None):
        """Copies python values into matlab-land.

        Parameters
        ----------
        varnames: string or list of strings
          the name(s) of the variables in matlab land

        values: python value or list of python values
          the values of the parameters in python land

        eng: A MatlabEngine instance of None
          if None, the default engine (set via the constructor) will be used

        int2float: boolena or None
          if True, int values (either scalar or numpy arrays) will be transferred as doubles
          if None, the default value (set via the constructor) will be used
          Note that this is brittle at the moment; for example, it won't convert ints in lists or tuples

        Returns
        -------
        A EngineVar variable list linked to varnames with the proper values in matlab land.
        """
        int2float = self._int2float if int2float is None else int2float
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
        if int2float:
            values = ints2floats(*values)
        # deal with nested matlab sequences
        values = map(_flatten_matlab_sequences, values)
        # treat differently enginevar from other values
        not_ev_names, not_ev_values, ev_names, ev_values = _segregate_evs(varnames, values)
        # treat differently matlab sequences from other values
        not_ev_names, not_ev_values, ms_names, ms_values = _segregate_mss(not_ev_names, not_ev_values)
        # values not in matlab land get transfered...
        if len(not_ev_names) > 0:
            self._put_hook(not_ev_names, not_ev_values, eng, int2float)
        # ...as opposed to values in matlab land, that get aliased (but careful with WOC)
        # and matlab-sequences, that get transferred as string
        if len(ev_names) > 0 or len(ms_names) > 0:
            command = ';'.join('%s=%s' % (alias, py2matstr(val)) for alias, val
                               in zip(ev_names, ev_values) + zip(ms_names, ms_values))
            eng.run_command(command, outs2py=False)
        # return a single var...
        if len(varnames) == 1:
            return EngineVar(eng, varnames[0])
        # ...or a list of vars
        return [EngineVar(eng, varname) for varname in varnames]

    def _put_hook(self, names, values, eng, int2float):
        raise NotImplementedError()


# --- Engines: one-stop shop for manage matlab <-> python bridges

class PyopyEngine(object):
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

    def __init__(self, transplanter=None, engine_location=None, warmup=False, num_threads=1):
        super(PyopyEngine, self).__init__()
        if transplanter is None:
            from pyopy.backend_oct2py import Oct2PyTransplanter
            self.transplanter = Oct2PyTransplanter(engine=self, tmp_prefix=self.__class__.__name__)
        else:
            self.transplanter = transplanter
        self._matlab_vars = {}
        self._num_results = 0
        self._num_variable_contexts = 0
        self._engine_location = engine_location
        self._num_threads = num_threads
        self._session = None
        if warmup:
            self.session()
            self.warmup()

    def warmup(self):
        """Initializes the engine to avoid delays in posterior calls."""
        self.run_command('ones(1);')

    def is_octave(self):
        """Returns True iff this engine is attached to octave."""
        raise NotImplementedError()

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

    def put(self, varnames, values, int2float=None):
        return self.transplanter.put(varnames, values, self, int2float=int2float)

    def get(self, varnames):
        return self.transplanter.get(varnames, self)

    def session(self):
        """Returns the backend session, opening it if necessary."""
        if self._session is None:
            self._session = self._session_hook()
            if not self.is_octave() and self._num_threads is not None and self._num_threads > 0:
                set_max_matlab_threads(self, self._num_threads)
        return self._session

    def _session_hook(self):
        raise NotImplementedError()

    def close_session(self):
        """Closes the backend session, if it is opened, shuttind down the engine."""
        try:
            self._close_session_hook()
        finally:
            self._session = None

    def _close_session_hook(self):
        raise NotImplementedError()

    def clear(self, varnames):
        """Cleans variables (string or EngineVariable instances) from the matlab/octave workspace."""
        if isinstance(varnames, (unicode, basestring, EngineVar)):
            varnames = [varnames]
        varnames = [var.name if isinstance(var, EngineVar) else var for var in varnames]
        if len(varnames) > 0:
            self.run_command('clear %s' % ' '.join(varnames), outs2py=False)

    def who(self, in_global=False):
        """Returns a list with the variable names in the current matlab/octave workspace."""
        # Remember also whos, class, clear, clearvars...
        if in_global:
            return self.run_command('ans=who(\'global\');', outs2py=True)[1][0]  # run with ()
        return self.run_command('ans=who();', outs2py=True)[1][0]  # run with ()

    def list_variables(self, in_global=False):
        """who() synonym."""
        return self.who(in_global)

    def max_comp_threads(self):
        return get_max_matlab_threads(self)

    def engine_class(self, name):
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


def set_max_matlab_threads(eng, n_threads=1):
    """Sets the maximum number of computational threads to use by a matlab engine.

    Note that we use the long-time deprecated but never disaapearing maxNumCompThreads,
    which can still lead to underlying libraries to keep using more than one computational thread. See:
      - http://www.mathworks.com/help/matlab/ref/maxnumcompthreads.html
      - http://www.mathworks.com/matlabcentral/answers/158192-maxnumcompthreads-hyperthreading-and-parpool
      - http://www.mathworks.com/matlabcentral/answers/
      83696-inconsistent-answers-from-different-computers-using-eig-function
      - http://www.mathworks.com/matlabcentral/answers/
      80129-definitive-answer-for-hyperthreading-and-the-parallel-computing-toolbox-pct

    We should prefer -singleCompThread when possible, but the Mathworks python engine does not support
    tweaking the command line arguments until the 2015a release. Probably by using that command line
    matlab can tweak too the number of threads of third party library (mkl & co?)

    As a side note, beyond the usual differences that happen with blas-calling soft,
    the differences in results between machines with different numbers of cores can
    sometimes be quite funny. See the initial benchmarks between machines with 4 cores
    (strall, str22, noisy) and strz (with 12 cores). Probably this is because of different order of
    rounding errors...
    """
    if eng.is_octave():
        raise ValueError('Engine should be over Matlab, not Octave')
    if n_threads is None or n_threads < 1:
        eng.run_command("maxNumCompThreads('auto')")
    eng.run_command("maxNumCompThreads(%d)" % n_threads)


def get_max_matlab_threads(eng):
    return 1 if eng.is_octave() else eng.run_function(1, 'maxNumCompThreads')


# --- Registry engines and default implementations

class PyopyEngines(object):

    _engines = {}
    _default = None

    @staticmethod
    def engine(name, thunk=None):
        if name not in PyopyEngines._engines:
            PyopyEngines._engines[name] = thunk()
        return PyopyEngines._engines[name]

    @staticmethod
    def engine_or_matlab_or_octave(engine='octave'):
        if engine is None:
            engine = 'octave'
        if isinstance(engine, PyopyEngine):  # should just quack like...
            return engine
        if isinstance(engine, basestring):
            if engine not in ('octave', 'matlab'):
                raise ValueError('engine string "%s" not allowed, only ("matlab" or "octave")')
            engine = engine == 'octave'
        return PyopyEngines.octave() if engine else PyopyEngines.matlab()

    @staticmethod
    def matlab():
        from pyopy.backend_mathworks import MathworksEngine
        return PyopyEngines.engine('matlab', thunk=MathworksEngine)

    @staticmethod
    def octave():
        from pyopy.backend_oct2py import Oct2PyEngine
        return PyopyEngines.engine('octave', thunk=Oct2PyEngine)

    @staticmethod
    def close(name):
        if name in PyopyEngines._engines:
            try:
                PyopyEngines._engines[name].close_session()
            except:
                pass
            finally:
                del PyopyEngines._engines[name]

    @staticmethod
    @atexit.register
    def close_all():
        for _, v in PyopyEngines._engines.iteritems():
            try:
                v.close_session()
            except:
                pass
        PyopyEngines._engines = {}

    @staticmethod
    def close_matlab():
        PyopyEngines.close('matlab')

    @staticmethod
    def close_octave():
        PyopyEngines.close('octave')

    @staticmethod
    def default():
        try:
            print 'Using matlab...'
            return PyopyEngines.matlab()
        except:
            print 'Using octave...'
            return PyopyEngines.octave()

    @staticmethod
    def set_default(name, thunk):
        raise NotImplementedError('To implement ASAP')
