PYOPY: PYthon -> Octave/matlab -> PYthon
========================================

Pyopy helps using matlab/octave libraries from python.
It provides functionality to parse matlab code and express calls to matlab functions
as calls to python functions. It also provides a generic mechanism to transfer data to
and run code in octave/matlab instances. Pyopy was born to provide python bindings to the 
[HCTSA time-series feature extraction library](http://www.comp-engine.org/timeseries/).
You can check-out this [quick demo](https://asciinema.org/a/18771).


Installation
------------

Pyopy is tested on linux and mac. It requires python 2.7, octave and/or matlab and the following python dependencies:
 
 - numpy
 - scipy
 - pandas
 - joblib
 - argh
 - whatami
 - lockfile
 
Pyopy design decouples command dispatching and data transfer.
Pyopy relies upon the excellent [oct2py](http://blink1073.github.io/oct2py/)
(of which a [slightly modified version](https://github.com/sdvillal/oct2py) is provided with pyopy).
for generic data transfers. For communicating with octave we recommend to fully install oct2py.
For communicating with matlab, we recommend the python matlab engine (available with matlab since
version 2014b). If you have an earlier matlab version, a slower and less tested backend
based on [pymatbridge](https://github.com/arokem/python-matlab-bridge) is also available.

### Install example using a [conda environment](http://conda.io/)

 - Install dependencies.
```sh
conda install numpy scipy pandas joblib
pip install argh whatami lockfile
```

 - To use matlab, install the [python matlab engine](http://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html). 

```sh
cd /opt/matlab/extern/engines/python  # if matlab is installed in /opt/matlab
python setup.py install  # unfortunately, pip would fail here
```

 - To use octave, install oct2py.

```sh
pip install oct2py
```

 - *pip install* pyopy in your environment.

```sh
# You can install the last release from pypi
pip install pyopy

# Alternatively, you can install master directly from github
pip install https://github.com/strawlab/pyopy/tarball/master


# Finally, you could also install a development version
cd wherever
git clone https://github.com/strawlab/pyopy.git
pip install -e pyopy
```

  - If using HCTSA: install and tweak pyopy's internal HCTSA copy.

```sh

# For HCTSA, using matlab is highly recommended.

 hctsa-cli install --engine matlab --force-download --generate-bindings

#
# This command will download or clone HCTSA from github into
#   ~/.pyopy/toolboxes/hctsa/
# Then it will patch it and mex extensions using either matlab or octave.
# This version of HCTSA will be dynamically added to the matlab/octave
# path each time "hctsa.prepare()" is called.
#

#
# The command will also regenerate the python bindings.
# This means regenerating the module:
#   "pyopy/hctsa/hctsa_bindings.py"
# Note that for the bindings to be regenerated succesfully,
# "hctsa-cli" needs write permissions on the directory where
# pyopy has been installed.
# This should not be a problem if you used "pip install -e" or
# if installed to a conda/virtualenv on your user dir.
#


# If you want to generate the mexes for the other engine, run:

 hctsa-cli install --engine octave

#
# Note not "--force-download" (otherwise the mex extensions for the
# first engine would be removed) nor ""--generate-bindings"
# (these need to be generated only once) should be specified.
# The same effect can be achieved running this command:
#   hctsa-cli install --engine all --force-download --generate-bindings
#

# As a final note, mexing can be redone when changing matlab/octave by:
 hctsa-cli mex --engine matlab
```

  - If all has gone well, this should work

```sh
hctsa-cli summary

Number of operators (functions in mfiles):    164
Number of operations (function + parameters): 1057
Number of features (operation + outvalue):     7778
Functions without operation: ['CO_TSTL_amutual', 'DN_Cumulants', 'DN_nlogL_norm', 'IN_AutoMutualInfo', 'IN_Initialize_MI', 'IN_MutualInfo', 'MF_GP_LearnHyperp', 'MF_ResidualAnalysis', 'NL_CaosMethod', 'PP_PreProcess', 'SB_CoarseGrain', 'SD_MakeSurrogates', 'TSTL_predict']
Operations without functions: ['DK_TheilerQ', 'DK_crinkle_statistic', 'DK_timerev_1', 'DK_timerev_2', 'DK_timerev_3', 'DK_timerev_4']
    (these are probably calls into other toolboxes)
Features without operations: []
    (usually this should be empty)

ipython
```

```python
In [1]: import numpy as np

In [2]: from pyopy.hctsa import hctsa

In [3]: _ = hctsa.prepare(engine='matlab')
Starting engine
Warming up
Configuring HCTSA
Setting up HCTSA operators
Hooray, we can use HCTSA now...

In [4]: hctsa.operations.AC_1(np.arange(100))
Out[4]: 0.96999999999999975
```


Licenses
--------

### Code distributed with pyopy

 - PYOPY itself: Modified BSD
 - OCT2PY: Modified BSD
 - OMPC: Modified BSD
 
### Optional matlab toolboxes

#### HCTSA

HCTSA must be downloaded separately (preferably using the *pyopy/hctsa/hctsa_install.py* script).
"Operators" are GPL licensed. Pyopy bindings rely on operators only.
Other parts of HCTSA are licensed as [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/). 
Other licenses in 3rd party code used by HCTSA (TOOLBOXES directory), mostly GPL, might apply. 
