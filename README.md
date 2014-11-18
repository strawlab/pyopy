# Installation

## Requirements

### Octave and the requirements or matlab and the requirements
### As for python: oct2py, numpy, scipy
### HCTSA: symlink or copy to blah

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

 - HCTSA: GPL


# Usage examples
