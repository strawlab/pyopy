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

 - Clone pyopy and pip-install it in your environment (releases coming sometime soon).

```sh
git clone https://github.com/strawlab/pyopy.git
cd pyopy
pip install -e .
```

  - If using HCTSA: install and tweak pyopy's internal HCTSA copy.

```sh
# The following command needs git installed and matlab/octave mex working.
# It will clone HCTSA from github into ~/.pyopy, patch it and mex extensions.
# For HCTSA, using matlab is highly recommended

 hctsa-cli install --engine matlab --force-download --generate-bindings

# Note that for the bindings to be regenerated succesfully hctsa-cli needs
# write permissions on the directory where pyopy has been installed
# (this should not be a problem if you used "pip install -e" or if installed
# to a conda/virtualenv on your user)
```

  - If all has gone well, this should work

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
