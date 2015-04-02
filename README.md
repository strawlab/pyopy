PYOPY: PYthon -> Octave/matlab -> PYthon
========================================

Pyopy helps using matlab/octave libraries from python.
It provides functionality to parse matlab code and express calls to matlab functions
as calls to python functions. It also provides a generic mechanism to transfer data to
and run code in octave/matlab instances. Pyopy was born to provide python bindings to the 
[HCTSA time-series feature extraction library](http://www.comp-engine.org/timeseries/).


Features
--------

Installation
------------

Pyopy is only tested on linux. It requires python 2.7 and the following python dependencies:
 
 - numpy
 - scipy
 - pandas
 - joblib
 - argh
 - whatami
 - lockfile
 
Optionally, some libraries can be used for data transfer and command pimping:

  - oct2py
  - python matlab bridge
  - matlab_wrapper

If you use conda/anaconda 

```sh
conda install numpy scipy pandas
pip install joblib argh whatami lockfile
```

Pyopy design decouples command dispatching and data transfer.
Pyopy relies upon the excellent [oct2py](http://blink1073.github.io/oct2py/) 
(of which a [slightly modified version](https://github.com/sdvillal/oct2py) is provided with pyopy).
for generic data transfers. For communicating to octave 

#
# To compile the pymatbridge octave messenger:
#   mkoctfile --mex -lzmq messenger.c
# Put somewhere in the octave path
# And for matlab:
#   /home/santi/Utils/Science/matlab/bin/mex -lzmq messenger.c
#

# Setup the matlab<->python bridges

The cleanest way to make python-matlab-bridge work with anaconda's libzmq 
is to use patchelf (as suggested in the python-matlab-bridge documentation).
Redefining LD_LIBRARY_PATH is another option, but then it is wise to
put in front of anaconda's lib dir other system libraries directories,
as to avoid anaconda libraries to screw-up other programs (like matlab).

https://github.com/arokem/python-matlab-bridge
https://github.com/NixOS/patchelf

Building patchelf is an option, there are also versions in anaconda/binstar
which work great and it is in the main repos of arch (so no need to really
bother installing on host).

patchelf --set-rpath /home/santi/Utils/Science/anaconda/lib messenger.mex
patchelf --set-rpath /home/santi/Utils/Science/anaconda/lib messenger.mexa64

## Linux 64

Installation in linux 64 and Matlab/Octave compatibility state:

  MEXing:
     - mex fails with the TSTool, the rest seem to be alright
       It is explicitly said it won't work with octave
         http://www.physik3.gwdg.de/tstool/HTML/node84.html
      But octave has made good progress on OO support, so I started giving it a try
      Not too seriously and unsuccessful after 1 hour,
      see TSTOOL-octave.tar.gz saved somewhere if ever we feel like keeping going on

  Notes on some of the missing toolboxes:
    - Octave misses a (compatible) Curve Fitting Toolbox
      See: http://www.krizka.net/2010/11/01/non-linear-fitting-using-gnuoctave-and-leasqr/


# Licenses

 - OMPC: GPL
 - OCT2PY: BSD
 - HCTSA: GPL
