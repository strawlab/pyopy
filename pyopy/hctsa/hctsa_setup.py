# coding=utf-8
"""Fix, install, mex HCTSA in octave/matlab land."""
import os
import os.path as op
import shutil
from subprocess import check_call
import urllib
import tarfile
from pyopy.base import PyopyEngines

from pyopy.hctsa import HCTSA_DIR, HCTSA_TOOLBOXES_DIR
from pyopy.code import rename_matlab_func
from pyopy.misc import ensure_dir, cd


def _download_hctsa(force=False, release_or_branch='OperationChanges', use_git=True):
    """Downloads HCTSA from github."""
    if op.isdir(HCTSA_DIR) and not force:
        return
    print 'Removing current installation...'
    shutil.rmtree(HCTSA_DIR, ignore_errors=True)
    if not use_git:
        url = 'https://github.com/SystemsAndSignalsGroup/hctsa/archive/%s.tar.gz' % release_or_branch
        tar = op.join(op.dirname(HCTSA_DIR), 'hctsa-%s.tar.gz' % release_or_branch)
        print 'Downloading %s...' % url
        urllib.urlretrieve(url, tar)  # Note: this can only work if the repo is public or if we bring auth into python
        print 'Decompressing %s...' % tar
        with tarfile.open(tar, 'r:gz') as tfile:
            tfile.extractall(op.dirname(tar))
        print 'Deleting %s...' % tar
        os.remove(tar)
        print 'Done'
    else:
        url = 'git@github.com:SystemsAndSignalsGroup/hctsa.git'
        check_call(['git clone %s %s' % (url, HCTSA_DIR)], shell=True)
        with cd(HCTSA_DIR):
            check_call(['git checkout %s' % release_or_branch], shell=True)


def _fix_fnames():
    """Fixes functions that do not correspond to their file name."""
    for mfile, wrong_funcname in {'SB_MotifThree.m': 'ST_MotifThree'}.iteritems():
        rename_matlab_func(op.join(HCTSA_DIR, 'Operations', mfile), wrong_funcname)


def _fix_shadowing():
    """Renames functions and scripts to avoid shadowing."""

    # Damn matlab lame lack of proper namespaces
    # more automatic stuff can be achieved (see e.g. externals/matlab-renamer) but this is quick and easy

    # Toolboxes/OpenTSTOOL/mex-dev/NN/TestSuite/test.m -> Toolboxes/OpenTSTOOL/mex-dev/NN/TestSuite/test_NN.m
    if op.isfile(op.join(HCTSA_TOOLBOXES_DIR, 'OpenTSTOOL', 'mex-dev', 'NN', 'TestSuite', 'test.m')):
        os.rename(op.join(HCTSA_TOOLBOXES_DIR, 'OpenTSTOOL', 'mex-dev', 'NN', 'TestSuite', 'test.m'),
                  op.join(HCTSA_TOOLBOXES_DIR, 'OpenTSTOOL', 'mex-dev', 'NN', 'TestSuite', 'test_NN.m'))

    # Toolboxes/OpenTSTOOL/mex-dev/Lyapunov/run.m -> Toolboxes/OpenTSTOOL/mex-dev/Lyapunov/run_lyapunov.m
    if op.isfile(op.join(HCTSA_TOOLBOXES_DIR, 'OpenTSTOOL', 'mex-dev', 'Lyapunov', 'run.m')):
        os.rename(op.join(HCTSA_TOOLBOXES_DIR, 'OpenTSTOOL', 'mex-dev', 'Lyapunov', 'run.m'),
                  op.join(HCTSA_TOOLBOXES_DIR, 'OpenTSTOOL', 'mex-dev', 'Lyapunov', 'run_lyapunov.m'))

    # unwrap.m from gpml -> unwrap_gpml.m
    # note this would change from one gpml release to other, so "ag 'unwrap\('" is your friend
    if op.isfile(op.join(HCTSA_TOOLBOXES_DIR, 'gpml', 'util', 'unwrap.m')):
        unwrap_calling_files = (
            ('inf', 'infPrior.m'),
            ('doc', 'usagePrior.m'),
            ('util', 'minimize_lbfgsb_objfun.m'),
            ('util', 'minimize_v2.m'),
            ('util', 'minimize.m'),
            ('util', 'unwrap.m'),
            ('util', 'minimize_v1.m'),
            ('util', 'minimize_lbfgsb_gradfun.m'),
            ('util', 'minimize_lbfgsb.m'),
        )
        for fp in unwrap_calling_files:
            fp = op.join(HCTSA_TOOLBOXES_DIR, 'gpml', *fp)
            with open(fp) as reader:
                text = reader.read().replace('unwrap(', 'unwrap_gpml(')
                with open(fp, 'w') as writer:
                    writer.write(text)
        os.rename(op.join(HCTSA_TOOLBOXES_DIR, 'gpml', 'util', 'unwrap.m'),
                  op.join(HCTSA_TOOLBOXES_DIR, 'gpml', 'util', 'unwrap_gpml.m'))


def _fix_includes():
    """
    Fix legacy and unnecessary matrix.h imports

    In recent versions of matlab, when building mex extensions it is not necessary anymore to import "matrix.h".
    (importing "mex.h" already informs of all the necessary declarations, at least on linux).
    Octave mex / mkoctfile goes one step further and refuses to build files importing "matrix.h".

    Here we just detect such cases and comment the line including matrix.h in HCTSA.
    """
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


def _fix_hctsa():
    """Applies some fixes to the HCTSA codebase so they work for us."""
    # N.B. running this function twice should not lead to any problem...
    _fix_fnames()
    _fix_shadowing()
    _fix_includes()


def _mex_hctsa(engine=None):
    """Compiles the mex extensions using the specified engine."""
    # At the moment only OpenTSTool fails to compile for Octave under Linux64 (as said in their docs)
    engine.run_command('cd %s' % HCTSA_TOOLBOXES_DIR)
    # Quick hack to stop OpenTSTOOL asking for user input
    mexext = engine.run_function(1, 'mexext')
    ensure_dir(op.join(HCTSA_TOOLBOXES_DIR, 'OpenTSTOOL', 'tstoolbox', 'mex', mexext))
    # mex all
    response, _ = engine.run_command('compile_mex')
    # feedback if available...
    print 'Compilation feedback:\n\t', response.stdout
    # We also need to compile TISEAN and put it in the PATH (see my PKGBUILD for arch)


def prepare_engine_for_hctsa(engine):
    """Loads HCTSA and octave dependencies in the engine."""
    # Adds HCTSA to the engine path, so it can be used
    engine.add_path(op.join(HCTSA_DIR, 'Operations'))
    engine.add_path(op.join(HCTSA_DIR, 'PeripheryFunctions'))
    engine.add_path(op.join(HCTSA_DIR, 'Toolboxes'))
    # Load dependencies from octave-forge
    # See also notes on optional package loading:
    # https://wiki.archlinux.org/index.php/Octave#Using_Octave.27s_installer
    if engine.is_octave():
        engine.run_command('pkg load parallel')
        engine.run_command('pkg load optim')
        engine.run_command('pkg load signal')
        engine.run_command('pkg load statistics')
        engine.run_command('pkg load econometrics')


def install_hctsa(engine='octave', force_download=False):
    """Fixes problems with the HCTSA codebase and mexes extensions.

    Parameters
    ----------
    engine: string or MatlabEngine, default 'octave'
      The engine to use to build the the mex files
      if 'octave' or 'matlab', the default engine will be used
      else it must quack like a MatlabEngine

    force_download: boolean, default False
      If true, HCTSA will be removed even if it already exists
    """
    # Download
    _download_hctsa(force=force_download)
    # Select the engine
    engine = PyopyEngines.engine_or_matlab_or_octave(engine)
    # Fix some problems with the codebase to allow compilation with octave and get rid of shadowing
    _fix_hctsa()
    # Add HCTSA to the engine path
    prepare_engine_for_hctsa(engine)
    # Build the mex files
    _mex_hctsa(engine)


if __name__ == '__main__':
    install_hctsa(engine='matlab', force_download=True)  # make this true to run from anew
    install_hctsa(engine='octave', force_download=False)
    print 'Done'
