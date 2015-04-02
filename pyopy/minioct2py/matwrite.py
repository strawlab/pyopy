"""
.. module:: matwrite
   :synopsis: Write Python values into an MAT file for Octave.
              Strives to preserve both value and type in transit.

.. moduleauthor:: Steven Silvester <steven.silvester@ieee.org>

Modified to work with pyopy by Santi Villalba.
"""
import os

from scipy.io import savemat
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

from pyopy.base import EngineVar
from pyopy.base import MatlabSequence
from pyopy.minioct2py.utils import Oct2PyError
from pyopy.minioct2py.compat import unicode


class MatWrite(object):
    """Write Python values into a MAT file for Octave.

    Strives to preserve both value and type in transit.
    """

    # Essentially the same as MatWrite in Oct2Py
    # Adds hacks to allow:
    #  - data already in (octave/matlab)-land to be reused (via EngineVar machinery)
    #  - support for MatlabSequence
    #  - support for pure numeric cells
    #  - WIP: support for nested cells (lists/tuples with lists/tuples as elements)

    # This is some copy&paste from oct2py version...
    __OCT2PY_COPY_PASTE_VER__ = 'f7aa89b909cbb5959ddedf3ab3e743898eac3d45'
    # this version changed a lot how temporaries are handled, test with HCTSA

    def __init__(self, oned_as='row', convert_to_float=True):
        self.oned_as = oned_as
        self.convert_to_float = convert_to_float
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
            # PYOPY: engine variable
            if isinstance(var, EngineVar):
                argin_list.append(var.name)
                continue
            # /PYOPY
            # PYOPY: matlab sequences
            if isinstance(var, MatlabSequence):
                argin_list.append(var.matlab_sequence_string())
                continue
            # /PYOPY
            if names:
                argin_list.append(names.pop(0))
            else:
                argin_list.append("%s__" % chr(ascii_code))
            # for structs - recursively add the elements
            try:
                if isinstance(var, dict):
                    data[argin_list[-1]] = putvals(var, self.convert_to_float)
                # PYOPY: numeric cells as cells
                elif isinstance(var, (list, tuple)) and 0 < len(var) and var[-1] == '_celltrick_':
                    data[argin_list[-1]] = putval(np.array(var[:-1], dtype=object), self.convert_to_float)
                # /PYOPY
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


def putvals(dict_, convert_to_float=True):
    """
    Put a nested dict into the MAT file as a struct

    Parameters
    ==========
    dict_ : dict
        Dictionary of object(s) to store
    convert_to_float : bool
        If true, convert integer types to float

    Returns
    =======
    out : array
        Dictionary of object(s), ready for transit

    """
    data = dict()
    for key in dict_.keys():
        if isinstance(dict_[key], dict):
            data[key] = putvals(dict_[key], convert_to_float)
        else:
            data[key] = putval(dict_[key], convert_to_float)
    return data


def putval(data, convert_to_float=True):
    """
    Convert data into a state suitable for transfer.

    Parameters
    ==========
    data : object
        Value to write to file.
    convert_to_float : bool
        If true, convert integer types to float.

    Returns
    =======
    out : object
        Object, ready for transit

    Notes
    =====
    Several considerations must be made
    for data type to ensure proper read/write of the MAT file.
    Currently the following types supported: float96, complex192, void

    """
    if data is None:
        data = np.NaN
    if isinstance(data, set):
        data = list(data)
    if isinstance(data, list):
        # PYOPY hack for nested cells to work, help with HCTSA but it is dodgy FIXME
        def nested_list_hack(el):
            if isinstance(el, list) and str_in_list(el):
                return np.array(el, dtype=np.object)
            return el
        data = map(nested_list_hack, data)
        # /PYOPY
        # hack to get a viable cell object
        if str_in_list(data):
            try:
                data = np.array(data, dtype=np.object)
            except ValueError as err:  # pragma: no cover
                raise Oct2PyError(err)
        else:
            out = []
            for el in data:
                if isinstance(el, np.ndarray):
                    cell = np.zeros((1,), dtype=np.object)
                    cell[0] = el
                    out.append(cell)
                elif isinstance(el, (csr_matrix, csc_matrix)):
                    out.append(el.astype(np.float64))
                else:
                    out.append(el)
            return out
    if isinstance(data, (str, unicode)):
        return data
    if isinstance(data, (csr_matrix, csc_matrix)):
        return data.astype(np.float64)
    try:
        data = np.array(data)
    except ValueError:  # pragma: no cover
        data = np.array(data, dtype=object)
    dstr = data.dtype.str
    if 'c' in dstr and dstr[-2:] == '24':
        raise Oct2PyError('Datatype not supported: {0}'.format(data.dtype))
    elif 'f' in dstr and dstr[-2:] == '12':
        raise Oct2PyError('Datatype not supported: {0}'.format(data.dtype))
    elif 'V' in dstr:
        raise Oct2PyError('Datatype not supported: {0}'.format(data.dtype))
    elif dstr == '|b1':
        data = data.astype(np.int8)
    elif dstr == '<m8[us]' or dstr == '<M8[us]':
        data = data.astype(np.uint64)
    elif '|S' in dstr or '<U' in dstr:
        data = data.astype(np.object)
    elif '<c' in dstr and np.alltrue(data.imag == 0):
        data.imag = 1e-9
    if data.dtype.name in ['float128', 'complex256']:
        raise Oct2PyError('Datatype not supported: {0}'.format(data.dtype))
    if data.dtype == 'object' and len(data.shape) > 1:
        data = data.T
    if convert_to_float and data.dtype.kind in 'uib':
        data = data.astype(float)
    return data


def str_in_list(list_):
    """See if there are any strings in the given list"""
    for item in list_:
        if isinstance(item, (str, unicode)):
            return True
        elif isinstance(item, list):
            if str_in_list(item):
                return True


#
# TODO: proper spec of what must be converted to what
#   Start by: http://blink1073.github.io/oct2py/source/conversions.html
# Specially hairy are cells. The '_celltrick_' stuff is quite bad, we should probably use numpy object arrays.
# That started from oct2py deciding that lists with strings are to be transferred as cells and my being in a hurry.
# To represent cells as np arrays with dtype object on binding generation seems like the way to go
# (although it would add mutable state to functions signatures)
# See:
#   http://docs.scipy.org/doc/scipy-0.14.0/reference/tutorial/io.html#
# matlabwrapper seems to do the right thing:
#   https://github.com/mrkrd/matlab_wrapper/blob/ace84696f79e90f04dc783d3a7efb533c4510f56/matlab_wrapper/matlab_session.py#L599-L614
#   https://github.com/mrkrd/matlab_wrapper/blob/ace84696f79e90f04dc783d3a7efb533c4510f56/tests/test_matlab.py#L297-L380
# What about pymatbridge and the mathworks engine?
#
