# coding=utf-8
import os
import os.path as op
from user import home
from pyopy.misc import ensure_dir

PYOPY_DIR = op.abspath(op.dirname(__file__))
PYOPY_INTERNAL_TOOLBOXES_DIR = op.abspath(op.join(PYOPY_DIR, 'externals', 'toolboxes'))

PYOPY_USER_DIR = op.join(home, '.pyopy')
PYOPY_EXTERNAL_TOOLBOXES_DIR = os.getenv('PYOPY_TOOLBOXES')
if PYOPY_EXTERNAL_TOOLBOXES_DIR is None:
    PYOPY_EXTERNAL_TOOLBOXES_DIR = op.join(PYOPY_USER_DIR, '.pyopy', 'toolboxes')
ensure_dir(PYOPY_EXTERNAL_TOOLBOXES_DIR)
