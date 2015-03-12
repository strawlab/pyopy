# coding=utf-8
"""Command-line access to some tools to manage HCTSA."""
import argh

from pyopy.hctsa.hctsa_bindings_gen import gen_python_bindings
from pyopy.hctsa.hctsa_setup import install_hctsa
from pyopy.hctsa.hctsa_catalog import hctsa_summary


if __name__ == '__main__':
    parser = argh.ArghParser()
    parser.add_commands([
        # matlab world installation
        install_hctsa,
        # python-land binding generation
        gen_python_bindings,
        # catalog
        hctsa_summary,
    ])
    parser.dispatch()
