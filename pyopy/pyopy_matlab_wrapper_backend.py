# coding=utf-8
"""Using matlab_wrapper as backend."""
from pyopy.base import MatlabTransplanter, MatlabEngine, EngineResponse


class MatlabWrapperTransplanter(MatlabTransplanter):

    def _get_hook(self, varnames, eng):
        if len(varnames) == 1:
            return eng.session().get(varnames[0])
        return map(eng.session().get, varnames)

    def _put_hook(self, names, values, eng, int2float):
        for name, value in zip(names, values):
            eng.session().put(name, value)


class MatlabWrapperEngine(MatlabEngine):

    def __init__(self,
                 transplanter=None,
                 engine_location='/opt/matlab',  # FIXME: better inference, default to None
                 warmup=False):
        """
        Note that engine_location must, in this case, be the directory where matlab is installed.
        For example:
          engine_location = '/opt/matlab'
        If None, matlab_wrapper will try to guess, but at the moment their strategy is brittle
        (look for a matlab executable, it will fail as soon as the executable is, for example, in /usr/bin)
        """
        super(MatlabWrapperEngine, self).__init__(transplanter, engine_location, warmup)
        self._session = None

    def is_octave(self):
        return False

    def _run_command_hook(self, command):
        try:
            self.session().eval(command)
            return EngineResponse(success=True, code=command), command
        except RuntimeError, e:
            return EngineResponse(success=False, code=command, stdout=str(e)), command

    def session(self):
        if self._session is None:
            import matlab_wrapper
            self._session = matlab_wrapper.MatlabSession(matlab_root=self._engine_location)
        return self._session

    def close_session(self):
        try:
            self._session.__del__()  # but do not rely on reference count
        finally:
            self._session = None  # maybe this could have been enough, or del self._session

if __name__ == '__main__':

    from pyopy.pyopy_oct2py_backend import Oct2PyTransplanter

    with MatlabWrapperEngine(engine_location='/opt/matlab',
                             transplanter=Oct2PyTransplanter()) as eng:
        x = eng.run_function(1, 'ones', 10000, 10000)
        print x.shape
        x = eng.put('x', 2, int2float=False)
        print x.matlab_class()
        x = eng.put('x', 2, int2float=True)
        print x.matlab_class()

    with MatlabWrapperEngine(engine_location='/opt/matlab',
                             transplanter=MatlabWrapperTransplanter()) as eng:
        x = eng.run_function(1, 'ones', 10000, 10000)
        print x.shape
        x = eng.put('x', 2, int2float=False)
        print x.matlab_class()
        x = eng.put('x', 2, int2float=True)
        print x.matlab_class()
