#! /usr/bin/env python2
# coding=utf-8

# Authors: Santi Villalba <sdvillal@gmail.com>
# Licence: BSD 3 clause

from setuptools import setup


setup(
    name='pyopy',
    license='BSD 3 clause',
    description='Tools to talk to matlab/octave libraries, using a hopefully simple model',
    version='0.1-dev',
    url='https://github.com/strawlab/pyopy',
    author='Santi Villalba',
    author_email='sdvillal@gmail.com',
    classifiers=[  # plagiarism from sklearn
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD 3 clause'
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Unix',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],
    requires=['numpy',
              'scipy',
              'matplotlib',
              'oct2py',
              'pymatbridge',
              'argh'],
    # extras_require={
    #     'matlab': ['pymatbridge']
    # },
    test_require=['pytest'],
    # And of course, HCTSA and other pimped matlab libraries
)
