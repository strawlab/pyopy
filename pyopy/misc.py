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


def ensure_writable_dir(path):
    """Ensures that a path is a writable directory."""
    def check_path(path):
        if not op.isdir(path):
            raise Exception('%s exists but it is not a directory' % path)
        if not os.access(path, os.W_OK):
            raise Exception('%s is a directory but it is not writable' % path)
    if op.exists(path):
        check_path(path)
    else:
        try:
            os.makedirs(path)
        except Exception:
            if op.exists(path):  # Simpler than using a file lock to work on multithreading...
                check_path(path)
            else:
                raise
    return path


def ensure_dir(path):
    return ensure_writable_dir(path)


def float_or_int(val_string):
    """Given a string we know it is a number, get the float or int it represents."""
    if '.' in val_string:
        return float(val_string)
    return int(val_string)


def ints2floats(*args):
    """Returns a list with args where integers have been converted to floats.
    In python, '2' is an int; in matlab, it is a double, and that can give problems with matlab's picky type system.
    """
    # this is now solved also in oct2py putval, but it is handy to have it standalone
    def int2float(x):
        if isinstance(x, int):
            return float(x)
        if hasattr(x, 'dtype') and x.dtype.kind in 'uib':
            return x.astype(float)
        return x
    return map(int2float, args)


def is_iterable(v):
    """Checks whether an object is iterable or not."""
    try:
        iter(v)
    except:
        return False
    return True


# --- Stuff to generate identifiers

def strings_generator(prefix='', suffix=''):
    from string import digits, ascii_uppercase, ascii_lowercase
    from itertools import product

    chars = digits + ascii_uppercase + ascii_lowercase

    for n in xrange(1, 1000):
        for comb in product(chars, repeat=n):
            yield prefix + ''.join(comb) + suffix


def some_strings(n, as_list=False, prefix='', suffix=''):
    from itertools import islice
    if as_list:
        return list(islice(strings_generator(prefix=prefix, suffix=suffix), n))
    return islice(strings_generator(), n)


# --- Find type of file-system (for linux only, might work in MAC)
#    http://stackoverflow.com/questions/908188/is-there-any-way-of-detecting-if-a-drive-is-a-ssd
#    for ramdisk and network file systems, parse maybe filesystem outputed by mount

