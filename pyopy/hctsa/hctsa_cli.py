# coding=utf-8
"""Command-line access to some tools to manage HCTSA."""
import argh

from pyopy.hctsa.hctsa_bindings_gen import gen_bindings
from pyopy.hctsa.hctsa_install import install, mex
from pyopy.hctsa.hctsa_catalog import summary


def main():
    parser = argh.ArghParser()
    parser.add_commands([
        # matlab world installation
        install,
        # mexing
        mex,
        # python-land binding generation
        gen_bindings,
        # catalog summary
        summary,
    ])
    parser.dispatch()

if __name__ == '__main__':
    main()
