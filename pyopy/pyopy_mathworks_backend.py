# coding=utf-8
"""Using the matlab engine for python as a backend for pyopy.
See: http://www.mathworks.com/help/matlab/matlab-engine-for-python.html
"""
import sys
from StringIO import StringIO
from pyopy.base import MatlabTransplanter, MatlabEngine, EngineResponse

# Note: unicode support will start to happen from 2015a release
# http://www.mathworks.com/help/matlab/release-notes.html


# --- Output redirection

class SimpleTee(StringIO):

    #
    # N.B. we cannot use cStringIO.StringIO in python 2 because of both:
    #  - it cannot be inherited
    #  - MatlabEngine does check type, and cStringIO.StringIO does not typecheck
    #

    def __init__(self, other=sys.stdout):
        StringIO.__init__(self)
        self._other = other

    def write(self, message):
        if self._other is not None:
            self._other.write(message)
        StringIO.write(self, message)


class MathworksTransplanter(MatlabTransplanter):

    def _get_hook(self, varnames, eng):
        varnames = map(str, varnames)  # no unicode for matlab
        if len(varnames) == 1:
            return eng.session().workspace[varnames[0]]
        return map(eng.session().workspace.__getitem__, varnames)

    def _put_hook(self, names, values, eng, int2float):
        for name, value in zip(names, values):
            name = str(name) if isinstance(name, unicode) else name
            value = str(value) if isinstance(value, unicode) else value
            eng.session().workspace[name] = value


class MathworksEngine(MatlabEngine):

    def __init__(self,
                 transplanter=None,
                 warmup=False,
                 verbose=False):
        self._session = None
        self._eval = None
        self.verbose = verbose
        super(MathworksEngine, self).__init__(transplanter, warmup)

    def is_octave(self):
        return False

    def _run_command_hook(self, command):
        # we will be using this once and again
        if self._eval is None:
            self._eval = self.session().eval
        # output capture
        # if instantiating this ever becomes bottleneck, do it once and go for more complex protocols
        stdout = SimpleTee(sys.stdout if self.verbose else None)
        stderr = SimpleTee(sys.stderr if self.verbose else None)
        try:
            self._eval(str(command), nargout=0, stdout=stdout, stderr=stderr)
            return EngineResponse(success=True,
                                  code=command,
                                  stdout=stdout.getvalue(),
                                  stderr=stderr.getvalue()), command
        except RuntimeError as error:
            return EngineResponse(success=False,
                                  code=command,
                                  stdout=stdout.getvalue(),
                                  stderr=stderr.getvalue(),
                                  exception=error), command
        #
        # It would be better to narrow down exception type to these documented,
        # but I fear they change between matlab versions:
        #   http://www.mathworks.com/help/matlab/apiref/matlab.engine.matlabengine-class.html
        #

    def session(self):
        if self._session is None:
            from matlab.engine import start_matlab
            self._session = start_matlab()
        return self._session

    def close_session(self):
        try:
            # self._session.quit()
            self._session.exit()
        finally:
            self._session = None
            self._eval = None


if __name__ == '__main__':

    from pyopy.pyopy_oct2py_backend import Oct2PyTransplanter

    with MathworksEngine(transplanter=Oct2PyTransplanter()) as eng:
        x = eng.run_function(1, 'ones', 10000, 10000)
        print x.shape
        x = eng.put('x', 2, int2float=False)
        print x.matlab_class()
        x = eng.put('x', 2, int2float=True)
        print x.matlab_class()

    with MathworksEngine(transplanter=MathworksTransplanter()) as eng:
        # do not even dare to go 10000 x 10000, it won't finish, as per 2015/02/11
        x = eng.run_function(1, 'ones', 1000, 10)
        print x
        x = eng.put('x', 2, int2float=False)
        print x.matlab_class()
        x = eng.put('x', 2, int2float=True)
        print x.matlab_class()
