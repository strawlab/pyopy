#!/usr/bin/env python2
# coding=utf-8

# Authors: Santi Villalba <sdvillal@gmail.com>
# Licence: BSD 3 clause

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(
    name='pyopy',
    license='BSD 3 clause',
    description='PYthon->Octave->PYthon: Tools to pythonize matlab/octave libraries',
    version='0.1.1-dev',
    url='https://github.com/strawlab/pyopy',
    author='Santi Villalba',
    author_email='sdvillal@gmail.com',
    packages=['pyopy',
              'pyopy.minioct2py',
              'pyopy.externals',
              'pyopy.externals.ompc',
              'pyopy.tests',
              'pyopy.hctsa',
              'pyopy.hctsa.tests'],
    entry_points={
        'console_scripts': [
            'hctsa-cli = pyopy.hctsa.hctsa_cli:main',
        ]
    },
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
    ],
    requires=['numpy',
              'scipy',
              'pandas',
              'joblib',
              'argh',
              'whatami',
              'lockfile'],

    extras_require={
        'oct2py': ['oct2py>=3.1.0'],
        'pymatbridge': ['pymatbridge>=0.4.3'],
        'matlab_wrapper': ['matlab_wrapper>=0.9.6'],
        'mathworks': [],  # matlab python engine (http://www.mathworks.com/help/matlab/matlab-engine-for-python.html)
    },
    tests_require=['pytest'],
)
