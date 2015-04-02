# coding=utf-8
"""Octave Oct2Py adaptor and hacks"""
import atexit
import shutil
import tempfile

from pyopy.minioct2py.utils import Struct
from pyopy.minioct2py.matread import MatRead
from pyopy.minioct2py.matwrite import MatWrite
from pyopy.base import PyopyTransplanter, PyopyEngine, EngineResponse, outputs_from_command


# --- Transplanter

class Oct2PyTransplanter(PyopyTransplanter):
    """File based data transit using the great oct2py type conversion machinery.

    Recommended to use this with SSDs or, for best speeds, use a RAM-based fs
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
        self._matwrite = MatWrite(oned_as=oned_as, convert_to_float=False)
        self._matread = MatRead()
        super(Oct2PyTransplanter, self).__init__(engine=engine, int2float=int2float)

    # --- Matlab -> Python

    def _get_hook(self, varnames, eng):
        # adapted from oct2py.pull (N.B. as side effect, varname gets empty, so copy)
        tempdir = self._create_tempdir()
        try:
            self._matread.create_file(tempdir)
            argout_list, save_line = self._matread.setup(len(varnames), list(varnames))
            eng.eval(save_line)
            data = self._matread.extract_file()
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
            eng.eval(load_line)
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

        def non_returning_expression(command):
            return 'javaaddpath' in command

        new_command = ('ans_pyopy = %s' % command) \
            if (outputs_from_command(command) == ['ans'] and not non_returning_expression(command)) \
            else command

        text_response = self.session().eval(new_command,
                                            verbose=self._verbose, timeout=self._timeout, log=self._log)

        return EngineResponse(success=True, code=command, stdout=text_response), new_command

    # --- Session management

    def _session_hook(self):

        from oct2py import Oct2Py

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
                self._writer = MatWrite(convert_to_float=False)
                # FIXME: lame, ask for pull request of EngineVar + MatWrite

        return Oct2PyNotAll(executable=self._engine_location)

    def _close_session_hook(self):
        self._session.close()


# --- manage oct2py opened octave instance
# oct2py is ATM always opening an octave instance (see oct2py.__init__)
# as we cannot assume it won't be used, we at least make sure it is closed when exiting
# it would be possible to make this lazy without breaking oct2py interface:
# http://stackoverflow.com/questions/1462986/lazy-module-variables-can-it-be-done


@atexit.register
def close_oct2py_octave():
    try:
        import oct2py
        oct2py.octave.close()
    except:
        pass
