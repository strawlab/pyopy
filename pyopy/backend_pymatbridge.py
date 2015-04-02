# coding=utf-8
import uuid

from pyopy.base import PyopyEngine, EngineResponse, PyopyTransplanter


class PyMatBridgeTransplanter(PyopyTransplanter):

    def _get_hook(self, varnames, eng):
        if len(varnames) == 1:
            return eng.session().get_variable(varnames[0])
        return [eng.session().get_variable(name) for name in varnames]

    def _put_hook(self, varnames, values, eng, int2float):
        for name, value in zip(varnames, values):
            eng.session().set_variable(name, value)
            if not int2float and isinstance(value, int):
                eng.eval('%s=int64(%s)' % (name, name))  # FIXME: or int32, if we are in a 32 bit system


class PyMatBridgeEngine(PyopyEngine):

    def __init__(self,
                 engine_location=None,
                 transplanter=None,
                 sid=None,
                 log=False,
                 octave=False,
                 warmup=False,
                 num_threads=1):
        self._id = 'pymatbridge-' + str(uuid.uuid1()) if sid is None else sid
        self._octave = octave
        self._num_threads = num_threads
        self._log = log
        if engine_location is None:
            engine_location = 'octave' if octave else 'matlab'
        super(PyMatBridgeEngine, self).__init__(transplanter=transplanter,
                                                engine_location=engine_location,
                                                warmup=warmup)

    def is_octave(self):
        return self._octave

    # --- Command running

    def _run_command_hook(self, command):
        return EngineResponse.from_pymatbridge_response(self.session().run_code(command)), command

    # --- Session management

    def _session_hook(self):
        import pymatbridge
        # N.B. capture_stdout=True in John's branch
        # N.B. Octave requires version > 0.3 (git when writing this)
        eng = pymatbridge.Octave if self._octave else pymatbridge.Matlab
        if self._octave:
            startup_ops = None
        else:
            startup_ops = ' -nodesktop -nodisplay' if self._num_threads != 1 else \
                ' -nodesktop -nodisplay -singleCompThread'
        session = eng(executable=self._engine_location,
                      id=self._id,
                      socket_addr='ipc:///tmp/%s' % self._id,
                      log=self._log,
                      startup_options=startup_ops)
        session.start()
        return session

    def _close_session_hook(self):
        try:
            self._session.stop()
        finally:
            self._session = None

# TODO: mixed transplanter, rules for faster approach (e.g. json for scalars, files for large matrices)
