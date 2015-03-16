# coding=utf-8
"""Octave Oct2Py adaptor and hacks"""
import atexit
import os
import shutil
import tempfile

from scipy.io import savemat

import oct2py
from oct2py import Oct2PyError, Oct2Py, Struct
from oct2py.matwrite import MatWrite, putvals, putval
from oct2py.matread import MatRead

from pyopy.base import PyopyTransplanter, PyopyEngine, EngineResponse, EngineVar, MatlabSequence
from pyopy.code import outputs_from_command

# --- manage oct2py opened octave instance
# oct2py is ATM always opening an octave instance (see oct2py.__init__)
# as we cannot assume it won't be used, we at least make sure it is closed when exiting
# it would be possible to make this lazy without breaking oct2py interface:
# http://stackoverflow.com/questions/1462986/lazy-module-variables-can-it-be-done


@atexit.register
def close_oct2py_octave():
    try:
        oct2py.octave.close()
    except:
        pass


# --- Copy & paste & hack stuff from oct2py

class MatWriteNotAll(MatWrite):
    """Essentially the same as MatWrite in Oct2Py
    Adds a hack to allow data already in (octave/matlab)-land to be reused
    """

    # This is some copy&paste from oct2py version...
    __OCT2PY_COPY_PASTE_VER__ = '3.1.0'  # this version changed a lot how temporaries are handled, test with HCTSA

    def __init__(self, oned_as='row', convert_to_float=False):
        super(MatWriteNotAll, self).__init__(oned_as, convert_to_float)
        self.in_file = None

    def create_file(self, temp_dir, inputs, names=None):
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
                    data[argin_list[-1]] = putvals(var, self.convert_to_float)
                else:
                    data[argin_list[-1]] = putval(var, self.convert_to_float)
            except Oct2PyError:
                raise
            ascii_code += 1
        self.in_file = os.path.join(temp_dir, 'writer.mat')
        try:
            savemat(self.in_file, data, appendmat=False,
                    oned_as=self.oned_as, long_field_names=True)
        except KeyError:  # pragma: no cover
            raise Exception('could not save mat file')
        load_line = 'load {0} "{1}"'.format(self.in_file,
                                            '" "'.join(argin_list))
        return argin_list, load_line


class Oct2PyNotAll(Oct2Py):

    def __init__(self,
                 executable=None,
                 logger=None,
                 timeout=None,
                 oned_as='row',
                 temp_dir=None,
                 convert_to_float=False):
        self._writer = None
        super(Oct2PyNotAll, self).__init__(executable, logger, timeout, oned_as, temp_dir, convert_to_float)

    def restart(self):
        super(Oct2PyNotAll, self).restart()
        self._writer = MatWriteNotAll(convert_to_float=False)
        # FIXME: lame, ask for pull request of EngineVar + MatWrite


# --- Transplanter

class Oct2PyTransplanter(PyopyTransplanter):
    """File based data transit using the great oct2py type conversion machinery.

    Recommened is to use this with SSDs or, for flying speeds, use a RAM-based fs
    (ramfs, tmpfs and the like).
    """

    def __init__(self,
                 engine=None,
                 tmp_dir_root=None,
                 tmp_prefix='pymatbridge_engine_',
                 oned_as='row',
                 int2float=True):
        self._tmpdir_root = tmp_dir_root
        self._tmp_prefix = tmp_prefix
        self._tempdirs = set()
        self._matwrite = MatWriteNotAll(oned_as=oned_as, convert_to_float=False)
        self._matread = MatRead()
        super(Oct2PyTransplanter, self).__init__(engine=engine, int2float=int2float)

    # --- Matlab -> Python

    def _get_hook(self, varnames, eng):
        # adapted from oct2py.pull (N.B. as side effect, varname gets empty, so copy)
        tempdir = self._create_tempdir()
        try:
            self._matread.create_file(tempdir)
            argout_list, save_line = self._matread.setup(len(varnames), list(varnames))
            eng.run_command(save_line)
            data = self._matread.extract_file(variables=varnames)
            if isinstance(data, dict) and not isinstance(data, Struct):
                return [data.get(v, None) for v in argout_list]
            else:
                return data
        finally:
            self._delete_tempdir(tempdir)

    # --- Python -> Matlab

    def _put_hook(self, varnames, values, eng, int2float):
        tempdir = self._create_tempdir()
        try:
            _, load_line = self._matwrite.create_file(tempdir, values, varnames)
            # matlab does not understand if it is quoted with "
            load_line = load_line.replace('"', "'")
            eng.run_command(load_line)
        finally:
            # it would be better if oct2py allowed to specify full path, instead of hardcoding the file name
            self._delete_tempdir(tempdir)

    # --- Temporary files management

    def _create_tempdir(self):
        tempdir = tempfile.mkdtemp(prefix=self._tmp_prefix, dir=self._tmpdir_root)
        self._tempdirs.add(tempdir)
        return tempdir

    def _delete_tempdir(self, tempdir):
        # it would be better if oct2py allowed to specify full path, instead of hardcoding the file name
        shutil.rmtree(tempdir, ignore_errors=True)
        self._tempdirs.remove(tempdir)

    # --- Context manager

    def __exit__(self, etype, value, traceback):
        map(self._delete_tempdir, self._tempdirs)


# --- Engine

class Oct2PyEngine(PyopyEngine):

    def __init__(self,
                 engine_location='octave',
                 transplanter=None,
                 verbose=False,
                 timeout=None,
                 log=True,
                 warmup=False):
        self._verbose = verbose
        self._timeout = timeout
        self._log = log
        super(Oct2PyEngine, self).__init__(transplanter=transplanter,
                                           engine_location=engine_location,
                                           warmup=warmup,
                                           num_threads=1)

    def is_octave(self):
        return True

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

    # --- Session management

    def _session_hook(self):
        return Oct2PyNotAll(executable=self._engine_location)

    def _close_session_hook(self):
        self._session.close()
