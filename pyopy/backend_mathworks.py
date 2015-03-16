# coding=utf-8
"""Using the matlab engine for python as a backend for pyopy.
See: http://www.mathworks.com/help/matlab/matlab-engine-for-python.html
"""
import sys
from StringIO import StringIO

from pyopy.base import PyopyTransplanter, PyopyEngine, EngineResponse

# Note: unicode support will start to happen from 2015a release
# http://www.mathworks.com/help/matlab/release-notes.html


# --- Output redirection

class _SimpleTee(StringIO):

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


class MathworksTransplanter(PyopyTransplanter):

    def _get_hook(self, varnames, eng):
        varnames = map(str, varnames)  # no unicode for matlab 2014b
        if len(varnames) == 1:
            return eng.session().workspace[varnames[0]]
        return map(eng.session().workspace.__getitem__, varnames)

    def _put_hook(self, varnames, values, eng, int2float):
        for name, value in zip(varnames, values):
            name = str(name) if isinstance(name, unicode) else name  # no unicode for matlab 2014b
            value = str(value) if isinstance(value, unicode) else value  # no unicode for matlab 2014b
            eng.session().workspace[name] = value


class MathworksEngine(PyopyEngine):

    def __init__(self,
                 transplanter=None,
                 warmup=False,
                 verbose=False,
                 num_threads=1):
        """
        Parameters
        ----------

        transplanter: PyopyTransplanter or None
          The strategy used to transfer data between matlab and python
          if None, then the an Oct2PyTransplanter (file based) will be used

        warmup: boolean, default False
          if True then the engine will be initialised immediatly

        verbose: boolean, default False
          if True, matlab output will be also spit out by the python process

        num_threads: int or None, default 1
          the maximum number of computational threads used in matlab land
          1 is highly recommended; it currently only maps to '-singleCompThread' matlab >=2015a

        """
        self.verbose = verbose
        self._eval = None
        super(MathworksEngine, self).__init__(transplanter=transplanter,
                                              engine_location=None,
                                              warmup=warmup,
                                              num_threads=num_threads)

    def is_octave(self):
        return False

    def _run_command_hook(self, command):
        # we will be using this once and again
        if self._eval is None:
            self._eval = self.session().eval
        # output capture
        # if instantiating this ever becomes bottleneck, do it once and go for more complex protocols
        stdout = _SimpleTee(sys.stdout if self.verbose else None)
        stderr = _SimpleTee(sys.stderr if self.verbose else None)
        try:
            self._eval(str(command), nargout=0, stdout=stdout, stderr=stderr)
            return EngineResponse(success=True,
                                  code=command,
                                  stdout=stdout.getvalue(),
                                  stderr=stderr.getvalue()), command
        except Exception as error:
            return EngineResponse(success=False,
                                  code=command,
                                  stdout=stdout.getvalue(),
                                  stderr=stderr.getvalue(),
                                  exception=error), command
        #
        # It would be better to narrow down the exception type to these documented:
        #   http://www.mathworks.com/help/matlab/apiref/matlab.engine.matlabengine-class.html
        # but I fear they change between matlab versions:
        #

    def _session_hook(self):
        from matlab.engine import start_matlab
        startup_ops = '-nodesktop -nodisplay' if self._num_threads != 1 else \
            '-nodesktop -nodisplay -singleCompThread'
        try:
            return start_matlab(startup_ops)
        except:
            return start_matlab()

    def _close_session_hook(self):
        try:
            # self._session.quit()
            self._session.exit()
        finally:
            self._eval = None


if __name__ == '__main__':

    from pyopy.backend_oct2py import Oct2PyTransplanter

    with MathworksEngine(transplanter=Oct2PyTransplanter()) as eng:
        x = eng.run_function(1, 'ones', 10000, 10000)
        print x.shape
        x = eng.put('x', 2, int2float=False)
        print x.engine_class()
        x = eng.put('x', 2, int2float=True)
        print x.engine_class()

    with MathworksEngine(transplanter=MathworksTransplanter()) as eng:
        # do not even dare to go 10000 x 10000, it won't finish (as per Matlab 2014b)
        #   http://www.mathworks.com/help/matlab/matlab_external/matlab-arrays-as-python-variables.html
        # hopefully one day...
        x = eng.run_function(1, 'ones', 1000, 10)
        print x
        x = eng.put('x', 2, int2float=False)
        print x.engine_class()
        x = eng.put('x', 2, int2float=True)
        print x.engine_class()
