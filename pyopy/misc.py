# coding=utf-8
"""The usual useful functions jumble."""
import os.path as op
import os


def ensure_python_package(path):
    """Tries to ensure that the path contains a valid python package (i.e. contains __init__.py)."""
    if not op.isdir(path):
        os.makedirs(path)
    if not op.isfile(op.join(path, '__init__.py')):
        open(op.join(path, '__init__.py'), 'w').close()


class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = newPath
        self.savedPath = None

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)