# coding=utf-8
"""Fix, install, mex HCTSA in octave/matlab land."""
import os
import os.path as op
import shutil
from subprocess import check_call
import urllib
import tarfile
import argh
from pyopy.base import PyopyEngines
from pyopy.hctsa.hctsa_bindings_gen import gen_bindings
from pyopy.hctsa.hctsa_config import HCTSA_DIR, HCTSA_TOOLBOXES_DIR, HCTSA_MOPS_FILE, HCTSA_OPS_FILE
from pyopy.code import rename_matlab_func
from pyopy.misc import ensure_dir, cd


def _download_hctsa(force=False, release_or_branch='v0.9', use_git=False):
    """Downloads HCTSA from github."""
    if op.isdir(HCTSA_DIR) and not force:
        return
    if op.isdir(HCTSA_DIR):
        print 'Removing current installation...'
        shutil.rmtree(HCTSA_DIR, ignore_errors=False)
    if not use_git:
        url = 'https://github.com/benfulcher/hctsa/archive/%s.tar.gz' % release_or_branch
        tar = op.join(op.dirname(HCTSA_DIR), 'hctsa-%s.tar.gz' % release_or_branch)
        print 'Downloading %s...' % url
        urllib.urlretrieve(url, tar)
        print 'Decompressing %s...' % tar
        with tarfile.open(tar, 'r:gz') as tfile:
            tfile.extractall(op.dirname(tar))
        print 'Deleting %s...' % tar
        os.remove(tar)
        print 'Renaming directories'
        try:
            os.rename(op.join(op.dirname(HCTSA_DIR), 'hctsa-%s' % release_or_branch),
                      op.join(op.dirname(HCTSA_DIR), 'hctsa'))
        except OSError:
            # github renames v0.9 to 0.9, try to see if this is the case
            os.rename(op.join(op.dirname(HCTSA_DIR), 'hctsa-%s' % release_or_branch[1:]),
                      op.join(op.dirname(HCTSA_DIR), 'hctsa'))
        print 'Done'
    else:
        url = 'https://github.com/benfulcher/hctsa.git'
        check_call(['git clone %s %s' % (url, HCTSA_DIR)], shell=True)
        with cd(HCTSA_DIR):
            check_call(['git checkout "%s"' % release_or_branch], shell=True)


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
        'Max_Little/fastdfa/ML_fastdfa_core.c',
        'Max_Little/rpde/ML_close_ret.c'
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


def _fix_mops_ops():
    """Replaces '-' by '_' in operation names, and apply other fixes to mops."""

    REPLACEMENTS = (
        ("CO_FirstMin(y,'mi-kraskov2',3)", "CO_FirstMin(y,'mi-kraskov2','3')"),
        ('CO_FirstMin_mi-', 'CO_FirstMin_mi_'),
        ('EN_mse_1-10_2_015_diff1', 'EN_mse_1_10_2_015_diff1'),
        ('EN_mse_1-10_2_015_rescale_tau', 'EN_mse_1_10_2_015_rescale_tau'),
        ('EN_mse_1-10_2_015', 'EN_mse_1_10_2_015')
    )
    with open(HCTSA_MOPS_FILE) as mops, open(HCTSA_OPS_FILE) as ops:
        mops = mops.read()
        ops = ops.read()
        for bad, good in REPLACEMENTS:
            mops = mops.replace(bad, good)
            ops = ops.replace(bad, good)
        with open(HCTSA_MOPS_FILE, 'w') as mopsw, open(HCTSA_OPS_FILE, 'w') as opsw:
            mopsw.write(mops)
            opsw.write(ops)


def _fix_hctsa():
    """Applies some fixes to the HCTSA codebase so they work for us."""
    # N.B. running this function twice should not lead to any problem...
    _fix_fnames()
    _fix_shadowing()
    _fix_includes()
    _fix_mops_ops()


def _mex_hctsa(engine=None):
    """Compiles the mex extensions using the specified engine."""
    # At the moment only OpenTSTool fails to compile for Octave under Linux64 (as said in their docs)
    engine.eval('cd %s' % HCTSA_TOOLBOXES_DIR)
    # Quick hack to stop OpenTSTOOL asking for user input
    mexext = engine.run_function(1, 'mexext')
    ensure_dir(op.join(HCTSA_TOOLBOXES_DIR, 'OpenTSTOOL', 'tstoolbox', 'mex', mexext))
    # mex all
    response, _ = engine.eval('compile_mex')
    # feedback if available...
    print 'Compilation feedback:\n\t', response.stdout
    #
    # We also need to compile TISEAN and put it in the PATH
    # See my PKGBUILD for arch:
    #   https://gist.github.com/sdvillal/98e300a439dbe089340d
    # Also note there is a "octave-tisean" toolbox.
    #   http://octave.sourceforge.net/tisean/
    #   https://aur.archlinux.org/packages/octave-tisean/
    # For ubuntu, not big deal either (just make sure gfortran is there):
    #   http://ubuntuforums.org/archive/index.php/t-2233905.html?s=f8da67a291ad38e4c2a533d35b9f8a80
    # What I do is not even to create a package (checkinstall would make it easy), but:
    #   sudo apt-get install gfortran
    #   export FC=gfortran  # in case there are others, like ifort
    #   cd $PYOPY_HOME/pyopy/externals/toolboxes/hctsa/Toolboxes/Tisean_3.0.1
    #   ./configure --prefix /usr/lib/tisean  # /usr/lib/tisean so it is similar to arch
    #   make clean
    #   make
    #   sudo mkdir -p /usr/lib/tisean/bin
    #   sudo make install
    #   # --- then add /usr/lib/tisean/bin to the PATH --- #
    #


def hctsa_prepare_engine(engine):
    """Loads HCTSA and octave dependencies in the engine."""
    # Adds HCTSA to the engine path, so it can be used
    engine.add_path(op.join(HCTSA_DIR, 'Operations'))
    engine.add_path(op.join(HCTSA_DIR, 'PeripheryFunctions'))
    engine.add_path(op.join(HCTSA_DIR, 'Toolboxes'))
    # Load dependencies from octave-forge
    # See also notes on optional package loading:
    # https://wiki.archlinux.org/index.php/Octave#Using_Octave.27s_installer
    if engine.is_octave():
        def maybe_load(pkg):
            try:
                engine.eval('pkg load %s' % pkg)
            except:  # FIXME: broad
                print 'Warning: cannot load octave package "%s", maybe install it?' % pkg
        maybe_load('parallel')
        maybe_load('optim')
        maybe_load('signal')
        maybe_load('statistics')
        maybe_load('econometrics')
    # Tweaks java classpaths
    try:
        engine.eval('javaaddpath(\'%s\');' % op.join(HCTSA_TOOLBOXES_DIR, 'infodynamics-dist', 'infodynamics.jar'))
    except:
        print 'Warning, could not add infodynamics to the %s path, is there java support in the engine?' % \
              ('octave' if engine.is_octave() else 'matlab')


def install(engine='matlab',
            force_download=False, use_git=False, version='v0.9',
            generate_bindings=False):
    """Fixes problems with the HCTSA codebase and mexes extensions.

    Parameters
    ----------
    engine : string or MatlabEngine, default 'octave'
      The engine to use to build the the mex files
      if 'octave' or 'matlab', the default engine will be used
      if 'all', installation will proceed on both matlab and octave
      else it must quack like a MatlabEngine

    force_download : boolean, default False
      If True, HCTSA will be removed even if it already exists

    version : string
      The version to HCTSA to download.
      It can be any commit ("master", "v0.9", "6684609e2d4670d84875f0e68d3197949d740fbd")

    use_git : boolean, default False
      If True, the HCTSA repository will be cloned.
      If False, HCTSA will be downloaded without the git repo.

    generate_bindings : boolean, default False
      If True, python bindings will be regenerated.
      Not needed if you will use the same HCTSA version used to generate the current bindings.
    """
    if engine == 'all':
        install(engine='matlab', force_download=force_download, generate_bindings=generate_bindings)
        install(engine='octave', force_download=False, generate_bindings=False)
        return
    # Download
    _download_hctsa(force=force_download, release_or_branch=version, use_git=use_git)
    # Select the engine
    engine = PyopyEngines.engine_or_matlab_or_octave(engine)
    # Fix some problems with the codebase to allow compilation with octave and get rid of shadowing
    _fix_hctsa()
    # Add HCTSA to the engine path
    hctsa_prepare_engine(engine)
    # Build the mex files
    _mex_hctsa(engine)
    # Generate the bindings
    if generate_bindings:
        print 'Generating python bindings...'
        gen_bindings()


if __name__ == '__main__':
    argh.dispatch_command(install)
    print 'Done'
