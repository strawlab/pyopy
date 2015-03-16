#!/usr/bin/env python2
# coding=utf-8

# Authors: Santi Villalba <sdvillal@gmail.com>
# Licence: BSD 3 clause

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import pyopy


setup(
    name='pyopy',
    license='BSD 3 clause',
    description='PYthon->Octave->PYthon: Tools to pythonize matlab/octave libraries',
    version=pyopy.__version__,
    url='https://github.com/strawlab/pyopy',
    author='Santi Villalba',
    author_email='sdvillal@gmail.com',
    packages=['pyopy',
              'pyopy.tests',
              'pyopy.hctsa',
              'pyopy.hctsa.tests'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD 3 clause'
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Unix',
        'Programming Language :: Python :: 2.7',
    ],
    requires=['numpy',
              'scipy',
              'pandas',
              'joblib',
              'oct2py',  # FIXME == 3.1
              'argh',
              'whatami',
              'lockfile'],

    extras_require={
        'pymatbridge': ['pymatbridge'],
        'matlab_wrapper': ['matlab_wrapper'],
    },
    tests_require=['pytest'],

    platforms=['linux'],
)
