import os.path as op
from pyopy.config import PYOPY_TOOLBOXES_DIR, PYOPY_DIR

# ---- Some paths

HCTSA_DIR = op.abspath(op.join(PYOPY_TOOLBOXES_DIR, 'hctsa'))  # where hctsa is
HCTSA_OPERATIONS_DIR = op.join(HCTSA_DIR, 'Operations')  # where the operators are
HCTSA_TOOLBOXES_DIR = op.join(HCTSA_DIR, 'Toolboxes')  # where the 3rd party toolboxes are
HCTSA_MOPS_FILE = op.join(HCTSA_DIR, 'Database', 'INP_mops.txt')  # funcname, parameters -> feature_cat
HCTSA_OPS_FILE = op.join(HCTSA_DIR, 'Database', 'INP_ops.txt')  # feature_cat -> featurecat.singlefeat labels
HCTSA_TESTTS_DIR = op.join(HCTSA_DIR, 'TimeSeries')  # where test time series are
HCTSA_BINDINGS_DIR = op.join(PYOPY_DIR, 'hctsa')  # where the generated files will be
HCTSA_BINDINGS_FILE = op.join(HCTSA_BINDINGS_DIR, 'hctsa_bindings.py')  # where the python functions will be