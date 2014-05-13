# coding=utf-8
"""Some oct2py goodies."""
import os

from oct2py import Oct2PyError, Oct2Py
from oct2py.matwrite import MatWrite, putvals, putval
from oct2py.utils import create_file
from scipy.io import savemat

from pyopy.matlab_utils import MatlabSequence, MatlabVar


#########################
# Allow data already in octave-land to be reused
#########################
# This is some copy&paste from oct2py 1.3.0
#########################

class MatWriteNotAll(MatWrite):

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
            if isinstance(var, MatlabVar):  # SANTI: one change
                argin_list.append(var.varname)
                continue
            if isinstance(var, MatlabSequence):  # SANTI: another change
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
            self.in_file = create_file()
        try:
            savemat(self.in_file, data, appendmat=False, oned_as='row')
        except KeyError:  # pragma: no cover
            raise Exception('could not save mat file')
        load_line = 'load {} "{}"'.format(self.in_file,
                                          '" "'.join(argin_list))

        return argin_list, load_line


class Oct2PyNotAll(Oct2Py):
    def restart(self):
        super(Oct2PyNotAll, self).restart()
        self._writer = MatWriteNotAll()  # TODO: lame, ask for pull request of MatlabVar + MatWrite