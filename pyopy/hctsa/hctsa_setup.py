# coding=utf-8
"""Fix, install, mex HCTSA in octave/matlab land."""
import os.path as op
from pyopy.hctsa import HCTSA_DIR, HCTSA_TOOLBOXES_DIR
from pyopy.matlab_utils import rename_matlab_func, Oct2PyEngine, PyMatBridgeEngine


def fix_hctsa():
    """Applies some fixes to the HCTSA codebase so they work for us.
    N.B. running this function twice should not lead to any problem...
    """
    # Functions that do not correspond to their file name
    for mfile, wrong_funcname in {'SB_MotifThree.m': 'ST_MotifThree'}.iteritems():
        rename_matlab_func(op.join(HCTSA_DIR, 'Operations', mfile), wrong_funcname)


def mex_hctsa(engine=None):
    """Compiles the mex extensions using the specified engine."""
    # Fix matrix imports
    #
    # In recent versions of matlab, when building mex extensions it is not necessary anymore to import "matrix.h".
    # (importing "mex.h" already informs of all the necessary declarations).
    # Octave mex / mkoctfile goes one step further and refuses to build files importing "matrix.h".
    #
    # Here we just detect such cases and comment the line including matrix.h in HCTSA.
    FILES_INCLUDING_MATRIX = (
        'OpenTSTOOL/mex-dev/Utils/mixembed.cpp',
        'gpml/util/lbfgsb/arrayofmatrices.cpp',
        'gpml/util/lbfgsb/lbfgsb.cpp',
        'Max_Little/steps_bumps_toolkit/ML_kvsteps_core.c',
        'Max_Little/steps_bumps_toolkit/ML_kvsteps_core.cpp',
        'Max_Little/fastdfa/ML_fastdfa_core.c'
    )

    def comment_matrix_import(path):
        import re
        with open(path) as reader:
            text = reader.read()
            new_text = re.sub(r'^#include "matrix.h"', '/*#include "matrix.h"*/', text, flags=re.MULTILINE)
            with open(path, 'w') as writer:
                writer.write(new_text)

    for path in FILES_INCLUDING_MATRIX:
        comment_matrix_import(op.join(HCTSA_TOOLBOXES_DIR, path))
    # At the moment only OpenTSTool fails to compile under Linux64
    # We will need to modify OpenTSTOOL/mex-dev/makemex.m
    engine.run_command('cd %s' % HCTSA_TOOLBOXES_DIR)
    engine.run_command('compile_mex')


def prepare_engine_for_hctsa(engine):
    """Loads HCTSA and octave dependencies in the engine."""
    # Adds HCTSA to the engine path, so it can be used
    engine.add_path(op.join(HCTSA_DIR, 'Operations'))
    engine.add_path(op.join(HCTSA_DIR, 'PeripheryFunctions'))
    engine.add_path(op.join(HCTSA_DIR, 'Toolboxes'))
    # Load dependencies from octave-forge
    # See also notes on optional package loading:
    # https://wiki.archlinux.org/index.php/Octave#Using_Octave.27s_installer
    if isinstance(engine, Oct2PyEngine):
        engine.run_command('pkg load parallel')
        engine.run_command('pkg load optim')
        engine.run_command('pkg load signal')
        engine.run_command('pkg load statistics')
        engine.run_command('pkg load econometrics')


def install_hctsa(engine='octave'):
    """Fixes problems with the HCTSA codebase and mexes extensions.

    Parameters
    ----------
    engine: string or MatlabEngine
        The engine to use to build the the mex files
        if 'octave', Oct2PyEngine will be used; if 'matlab', PyMatlabEngine will be used;
        else it must be a MatlabEngine
    """
    if engine is None or engine == 'octave':
        engine = Oct2PyEngine()
    elif engine == 'matlab':
        engine = PyMatBridgeEngine()
    # Fix some problems with the codebase
    fix_hctsa()
    # Add HCTSA to the engine path
    prepare_engine_for_hctsa(engine)
    # Build the mex files
    mex_hctsa(engine)
