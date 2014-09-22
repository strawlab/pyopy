# coding=utf-8
"""
Tools to talk to the "Highly Comparative (Comp-Engine) Time Series" matlab time-series processing toolbox.
See: http://www.comp-engine.org/
     http://www.comp-engine.org/timeseries/browse-operation-code-by-category

HCTSA is a big library of "Operations" together with some extra code to create a database of time-series features
Each "Operation" maps a time-series to some outputs, and each output ("Feature") is a scalar.
A "Category" represents a call to an "Operator" with certain parameters, producing one or more "Features".

Installation in linux 64 and Matlab/Octave compatibility state:

  Third party:
    - yaourt -S --noconfirm --needed octave-mkl octave-signal octave-parallel
                                     octave-optim octave-econometrics octave-statistics
      or in octave (nasty)
      pkg install -forge signal parallel optim econometrics statistics
      See also notes on optional package loading:
        https://wiki.archlinux.org/index.php/Octave#Using_Octave.27s_installer
    - tisean (from AUR):
      yaourt -S --noconfirm gcc-libs-multilib gcc-multilib tisean

  MEXing:
     - mex fails with the TSTool, the rest seem to be alright

  Notes on some of the missing toolboxes:
    - Octave misses a (compatible) Curve Fitting Toolbox
      See: http://www.krizka.net/2010/11/01/non-linear-fitting-using-gnuoctave-and-leasqr/
"""
import os.path as op
from glob import glob
from oscail.common.config import Configurable

from pyopy.config import PYOPY_TOOLBOXES_DIR, PYOPY_DIR
from pyopy.matlab_utils import rename_matlab_func, parse_matlab_funcdef, \
    parse_matlab_params, matlab_funcname_from_filename, Oct2PyEngine, PyMatlabEngine
from pyopy.misc import ensure_python_package
import inspect

###################################
# Some paths
###################################

HCTSA_DIR = op.abspath(op.join(PYOPY_TOOLBOXES_DIR, 'hctsa'))  # where hctsa is
HCTSA_OPERATIONS_DIR = op.join(HCTSA_DIR, 'Operations')  # where the operators are
HCTSA_TOOLBOXES_DIR = op.join(HCTSA_DIR, 'Toolboxes')  # where the 3rd party toolboxes are
HCTSA_MOPS_FILE = op.join(HCTSA_DIR, 'Database', 'INP_mops.txt')  # funcname, parameters -> feature_cat
HCTSA_OPS_FILE = op.join(HCTSA_DIR, 'Database', 'INP_ops.txt')  # feature_cat -> featurecat.singlefeat labels
HCTSA_TESTTS_DIR = op.join(HCTSA_DIR, 'TimeSeries')  # where test time series are
HCTSA_BINDINGS_DIR = op.join(PYOPY_DIR, 'hctsa', 'bindings')  # where the generated files will be
BINDINGS_FILE = op.join(HCTSA_BINDINGS_DIR, 'hctsa_bindings.py')  # where the python functions will be


###################################
# Catalog for the operations in HCTSA
###################################

class HCTSAFunction(object):
    """A main function in an mfile, parsed to its main components, from the HCTSA library.

    Parameters
    ----------
    mfile : string
        The path to the mfile that contains the function

    outstring : string
        The output on the function definition (for example "out" or "[a,b]")

    params : string list
        The list of parameter names for the function

    doc : string
        The documentation of the mfile (everything before the function definition)

    code : string
        The body of the function (everything after the function definition,
        possibly including other auxiliary functions).
    """
    def __init__(self,
                 mfile,
                 outstring,
                 params,
                 doc,
                 code):
        super(HCTSAFunction, self).__init__()
        self.mfile = op.abspath(mfile)
        self.funcname = matlab_funcname_from_filename(self.mfile)
        self.outstring = outstring
        self.params = params
        self.doc = doc
        self.code = code


class HCTSAFeature(object):
    """
    A feature in HCTSA parlance is a single scalar generated from
    the application of an Operator (function) to a time-series
    using concrete parameter values.

    Parameters
    ----------
    featname : string
        The feature name in the HCTSA database (e.g. "CO_HistogramAMI_1_std1_2.min")

    category : HCTSAFeatureCategory
        The category of this parameter (holds function and parameter values information)

    outname : string
        The name of the field in the matlab struct that is outputted by the matlab function
        or None if the function returns already a single scalar (e.g. "min")

    tags : string list
        The tags given to the feature in HCTSA, useful for categorization
    """
    def __init__(self,
                 featname,
                 category,
                 outname,
                 tags):
        super(HCTSAFeature, self).__init__()
        self.featname = featname
        self.category = category
        self.outname = outname
        self.tags = tags


class HCTSAFeatureCategory(object):
    """
    A feature category in HCTSA parlance is the combination of
    and Operator (function) and concrete parameter values.

    This is a "category" because, given that an operator can
    return several values, this application can potentially
    give rise to many different features.

    Parameters
    ----------
    catname : string
        The category name (e.g. "CO_CompareMinAMI_even_2_80")

    funcname : string
        The function name associated to this category (e.g. "CO_CompareMinAMI")

    param_values : list
        A list with the values for the function that define this category (as objects in python land)

    is_commented : bool
        Whether the category is commented in the HCTSA code
    """
    def __init__(self,
                 catname,
                 funcname,  # TODO: make this a HCTSAFunction
                 param_values,
                 is_commented):
        super(HCTSAFeatureCategory, self).__init__()
        self.catname = catname
        self.funcname = funcname
        self.param_values = param_values
        self.is_commented = is_commented
        self.features = []

    def add_feature(self, hctsa_feature):
        """Adds a feature to this category "known features".

        Parameters
        ----------
        hctsa_feature : HCTSAFeature
            A feature to add to this category (note that at the moment we allow duplicates)
        """
        self.features.append(hctsa_feature)


class HCTSACatalog(object):
    """
    Puts together the information of functions, parameter values and outputs from the HCTSA library.

    Parameters
    ----------
    mfiles_dir : path
        The directory where the matlab m files for the HCTSA operations reside.

    mops_file : path
        The path to the file where we find the map {category -> (function, param_values)}

    ops_file : path
        The path to the file where we find the map {feature_name -> (category, output_name)}

    Useful Members
    --------------
    functions_dict : dictionary
        A map {function_namer -> HCTSAFunction}

    categories_dict : dictionary
        A map {category_name -> HCTSAFeatureCategory}

    features_dict : dictionary
        A map {feature_name -> HCTSAFeature}
    """
    def __init__(self,
                 mfiles_dir=HCTSA_OPERATIONS_DIR,
                 mops_file=HCTSA_MOPS_FILE,
                 ops_file=HCTSA_OPS_FILE):
        super(HCTSACatalog, self).__init__()

        self.mfiles_dir = mfiles_dir
        self.mops_file = mops_file
        self.ops_file = ops_file

        self.functions_dict = None
        self.categories_dict = None
        self.features_dict = None

        self._build_hctsa_catalog()

    def _build_hctsa_catalog(self, ignore_commented=True):
        """Builds the catalog.
        A feature is a single number obtained by selecting a concrete output after a concrete call to one operation.
        To build the time-series database, HCTSA uses two files to declare parameter values and output values.
        """

        # Manage commented out operations, which are these starting with "#"
        def manage_comments(line):
            line = line.strip()
            if line.startswith('#'):  # Commented out operations
                if ignore_commented:
                    return None, True
                return line[1:].strip(), True
            return line, False

        #
        # 1) Get all the categories defined in the MOPS (metaoperations?) file
        #
        # A category is a concrete call to an operator with concrete parameters, and can be "commented"
        # (meaning is ignored by the HCTSA code).
        #
        # Because each call can return several many named features after the same computation,
        # these calls are grouped in FeatureCategories...
        #

        def parse_mops_line(line):
            """Parses a line of the MOPS file.

            An example line:
              'CO_CompareMinAMI(y,'even',[2:80])	CO_CompareMinAMI_even_2_80'

            Gets splitted as:
              category = 'CO_CompareMinAMI_even_2_80'
              funcname = 'CO_CompareMinAMI'
              params = ['even', '2:80']
              is_commented = False
            """
            line, is_commented = manage_comments(line)
            if line is None:  # Ignore commented lines
                return None, None, None, None
            callspec, category = line.split()
            funcname, _, params = callspec.partition('(')
            funcname = funcname.strip()
            params = params.rpartition(')')[0].strip()
            params = params.partition(',')[2]  # The first parameter is always the time series
            params = parse_matlab_params(params)
            return category, funcname, params, is_commented

        self.categories_dict = {}
        with open(self.mops_file) as reader:
            for line in reader:
                if not line.strip():
                    continue
                categoryname, funcname, params, is_commented = parse_mops_line(line)
                if categoryname is None:
                    continue  # Ignore commented lines
                if categoryname in self.categories_dict:
                    raise Exception('Repeated category: %s' % categoryname)
                self.categories_dict[categoryname] = HCTSAFeatureCategory(categoryname, funcname, params, is_commented)

        #
        # 2) Get all the features defined in the OPS (operations) file
        # A feature maps a category with the concrete scalar that needs to be taken out of the function call.
        #

        def parse_ops_line(line):
            """Parses a line of the OPS file.

            An example line:
              CO_CompareMinAMI_even_2_80.min	CO_CompareMinAMI_even_2_80_min	correlation,AMI

            Gets splitted as:
              category = 'CO_CompareMinAMI_even_2_80'
              outname = 'min'
              featname = 'CO_CompareMinAMI_even_2_80_min'
              tags = ['correlation', 'ami']
            """
            line, is_commented = manage_comments(line)
            if line is None:
                return None, None, None, None
            parts = line.split()
            outname, featname, tags = parts if 3 == len(parts) else (parts[0], parts[1], None)
            if '.' in outname:
                categoryname, outname = outname.split('.')  # Operator returns a struct
            else:
                categoryname, outname = outname, None  # Operator returns a scalar
            tags = tags.split(',') if tags is not None else None
            return categoryname, outname, featname, tags

        self.features_dict = {}
        with open(self.ops_file) as reader:
            for line in reader:
                if not len(line.strip()):
                    continue
                categoryname, outname, featname, tags = parse_ops_line(line)
                if categoryname is None:
                    continue  # Ignore commented lines
                category = self.categories_dict.get(categoryname,
                                                    HCTSAFeatureCategory(categoryname, None, None, None))
                feature = HCTSAFeature(featname, category, outname, tags)
                if featname in self.features_dict:
                    msg = 'Warning: the feature %s is defined more than once, ignoring...' % featname
                    print msg
                    continue  # But actually keep these that are not commented
                    # raise Exception(msg)
                self.features_dict[featname] = feature
                category.add_feature(feature)  # Other way round

        #
        # 3) Get all the operations defined in m-files.
        # The Operators library is very consistent:
        # - The signature has always the form "out = func(y,...)"
        #   - The documentation is consistently formatted
        #   - The default values are many times enforced the same way (but it is not useful)
        #   - They return either a single variable or a struct with many output features
        #     (and in few cases unpacked tuples)
        #
        # We can take advantage to generate python bindings without bothering with a more generic solution.
        #

        self.functions_dict = {}
        for mfile in sorted(glob(op.join(self.mfiles_dir, '*.m'))):
            doc, outstring, funcname, parameters, code = parse_matlab_funcdef(mfile)
            parameters = parameters[1:]  # The first parameter is always the time series
            doc = doc.split('% -------------------------------------')[2]  # Remove header and license
            self.functions_dict[funcname] = HCTSAFunction(mfile, outstring, parameters, doc, code)

    def report(self):
        report = [
            'Number of operators (functions in mfiles):    %d' % len(self.functions_dict),

            'Number of categories (function + parameters): %d' % len(self.categories_dict),

            'Number of features (category + outvalue):     %d' % len(self.features_dict),

            'Functions without categories: %s' % sorted(set(self.functions_dict.keys()) -
                                                        set(cat.funcname for cat in self.categories_dict.values())),

            'Categories without functions: %s' % sorted(cat.catname for cat in self.categories_dict.values() if
                                                        cat.funcname not in self.functions_dict),
            '    (these are probably calls into other toolboxes)',

            'Features without categories: %s' % sorted(fea.featname for fea in self.features_dict.values() if
                                                       fea.category.catname not in self.categories_dict),
            '    (usually this should be empty)',
        ]
        return '\n'.join(report)

    def default_parameters(self, funcname):
        """Returns default parameter values for a function as one (random)
        of the calls to generate features using such function.
        """
        for cat in self.categories_dict.values():
            if cat.funcname == funcname:
                return cat.param_values
        return []

    def function_parameter_values(self, funcname):
        """Returns all the values that a function takes to span a HTCSA category."""
        return [cat.param_values for cat in self.categories_dict.values() if cat.funcname == funcname]

    def function_outnames(self, funcname):
        """Returns all the outnames used in HCTSA to extract scalars
        from the application of this function to a timeseries.

        Note that [None] and [] have different meanings here:
          - [None] means the operation output is used as is in HCTSA
          - [] means the operation is not used at all un HCTSA
        """
        outnames = set()
        for category in self.categories_dict.itervalues():
            if category.funcname == funcname:
                for feature in category.features:
                    outnames.add(feature.outname)
        return sorted(outnames)


def hctsa_summary():
    """Shows a summary of the operations in the current HCTSA codebase."""
    print HCTSACatalog().report()


###################################
# Python bindings generation
###################################


class HCTSAFeatureExtractor(Configurable):

    def __init__(self):
        super(HCTSAFeatureExtractor, self).__init__(add_descriptors=False)

    def evaluate(self, engine, x):
        """Evaluates this feature for the time series x on the specified matlab engine."""
        pass

    def text_call(self, varname):
        """Represents a call to this function as a matlab expression."""
        pass

    def partial(self, engine):
        """Returns a MatlabVariable representing a partial application for this function and parameters."""
        pass

    def text_partial(self):
        """Represents a partial application of this function."""
        pass


def class_code_from_function(hctsa_function, outnames=None, function_prefix='HCTSA_'):

    # Our indentation levels
    indent1 = ' ' * 4
    indent2 = ' ' * 8

    # Introspection...
    func_name = hctsa_function.__name__
    mlab_name = func_name[len(function_prefix):]
    docstring = hctsa_function.__doc__
    args, _, _, defaults = inspect.getargspec(hctsa_function)

    # Arguments string
    if defaults is not None:
        args_string = ', '.join('%s=%r' % (name, value) for name, value in zip(args[2:], defaults))
    else:
        args_string = ''

    # Output values for the operator
    if outnames is None:
        outnames = HCTSACatalog().function_outnames(mlab_name)
    outnames_declaration = ''
    if len(outnames) > 1:
        outnames_declaration = '\n' + indent1 + 'outnames = (\'%s\',\n' % outnames[0]
        visual_indent = indent1 + ' ' * len('outnames = (')
        outnames_declaration += visual_indent + visual_indent.join('\'%s\',\n' % outname for outname in outnames[1:])
        outnames_declaration = outnames_declaration[:-2] + ')\n'

    # Constructor body
    constructor_string = indent2 + 'super(%s, self).__init__(add_descriptors=False)\n' % mlab_name
    if len(args) > 2:
        constructor_string += indent2 + indent2.join('self.%s = %s\n' % (param, param) for param in args[2:])

    # eval method
    if len(args) == 2:
        eval_method = indent1 + '@staticmethod\n'
        eval_method += indent1 + 'def eval(engine, x):\n'
        eval_method += indent2 + 'return %s(engine, x)' % func_name
    else:
        eval_method = indent1 + 'def eval(self, engine, x):\n'
        eval_method += indent2 + 'return %s(engine,\n' % func_name
        visual_indent = indent2 + ' ' * len('return %s(' % func_name)
        eval_method += visual_indent + 'x,\n'
        if len(args) > 3:
            eval_method += visual_indent + visual_indent.join('%s=self.%s,\n' % (param, param)
                                                              for param in args[2:-1])
        eval_method += visual_indent + '%s=self.%s)' % (args[-1], args[-1])

    code = [
        'class %s(Configurable):' % mlab_name,
        '%s"""%s"""' % (indent1, docstring),
        outnames_declaration,
        indent1 + 'def __init__(self, %s):' % args_string,
        constructor_string,
        eval_method,
        # '\n' + indent1 + 'def text_call(self, varname):',
        # indent2 + 'pass'
        # indent2 + 'return \'%s(\%s, %s)\' \% varname' % (mlab_name, 'FIXTHISWITHPY2MATLAB')
    ]
    return mlab_name, '\n'.join(code)


def gen_python_bindings(hctsa_catalog=None):
    """Generates the python bindings to the library operators."""

    def genfunction(funcname, parameters, doc, outnames, defaults=None, prefix='HCTSA'):
        """Generates python bindings for calling the feature extractors."""

        pyfuncname = '%s_%s' % (prefix, funcname)

        # avoid shadowing buitins with parameters...
        parameters = [{'ord': 'ord_'}.get(param, param)for param in parameters]
        # build the function signature
        if defaults is None:
            defaults = [None] * len(parameters)
        else:
            defaults.extend([None] * (len(parameters) - len(defaults)))
        parameter_string = '' if len(parameters) == 0 else \
            ', ' + ', '.join(['%s=%r' % (param, default) for param, default in zip(parameters, defaults)])
        # ...better tuples than lists...
        parameter_string = parameter_string.replace('[', '(').replace(']', ')')
        # def line
        defline = 'def %s(eng, x%s):' % (pyfuncname, parameter_string)
        # ...cosmetic for long lines
        if len(defline) > 120:
            one_middle_comma = defline.find(',', 80, 100) + 1
            visual_indent = ' ' * len('def %s(' % pyfuncname)
            defline = defline[:one_middle_comma] + '\n' + visual_indent + defline[one_middle_comma:].strip()

        # documentation
        indented_doc = '\n'.join(['    %s' % line for line in doc.splitlines()[1:]])
        doc_line = '\n'.join(['    """',
                              '    Matlab doc:',
                              '    %s' % ('-' * 40),
                              '%s' % indented_doc,
                              '    %s' % ('-' * 40),
                              '    """'])

        # out function
        outfunc = ['    def outfunc(out):']
        if not outnames or (len(outnames) == 1 and outnames[0] is None):
            outfunc.append('        return out')
        else:
            if len(outnames) == 1:
                outfunc.append('        return {outname: out[outname] for outname in %r}' % outnames)
            else:
                outfunc.append('        return {outname: out[outname] for outname in [\'%s\',' % outnames[0])
                visual_indent = ' ' * len('        return {outname: out[outname] for outname in [')
                for outname in outnames[1:]:
                    outfunc.append('%s\'%s\',' % (visual_indent, outname))
                outfunc[-1] = outfunc[-1][:-1] + ']}'
        outfunc_line = '\n'.join(outfunc)

        # function body
        body = []
        for i, parameter in enumerate(parameters):
            body.append('    %s %s is None:' % ('if' if i == 0 else 'elif', parameter))
            body.append('        out = eng.run_function(\'%s\', x, %s)' % (funcname, ', '.join(parameters[0:i])))
        if not parameters:
            body.append('    out = eng.run_function(\'%s\', x, %s)' % (funcname, ', '.join(parameters)))
        else:
            body.append('    else:')
            body.append('        out = eng.run_function(\'%s\', x, %s)' % (funcname, ', '.join(parameters)))
        body.append('    return outfunc(out)')
        body_line = '\n'.join(body)

        # assemble all together
        return pyfuncname, '\n'.join([defline, doc_line, outfunc_line, body_line])

    # Read-in the catalog
    if hctsa_catalog is None:
        hctsa_catalog = HCTSACatalog()
    # Ensure that the destination python package exists
    ensure_python_package(HCTSA_BINDINGS_DIR)
    with open(BINDINGS_FILE, 'w') as writer:
        # Bindings imports
        binding_imports = (
            'from pyopy.matlab_utils import MatlabSequence',
            'from oscail.common.config import Configurable')
        exec '\n'.join(binding_imports) in globals()  # We are using nasty execs around tha need these imports
        # Write the header
        writer.write('# coding=utf-8\n')
        writer.write('\n'.join(binding_imports) + '\n\n\n')
        # Write the functions and classes
        funcnames = []
        classnames = []
        for func in sorted(hctsa_catalog.functions_dict.values(), key=lambda f: f.funcname):
            funcname, funcdef = genfunction(func.funcname,
                                            func.params,
                                            func.doc,
                                            hctsa_catalog.function_outnames(func.funcname),
                                            hctsa_catalog.default_parameters(func.funcname))
            exec funcdef in globals()  # Nasty, but avoids second passes and acts as a sanity check
            classname, classdef = class_code_from_function(eval(funcname),
                                                           hctsa_catalog.function_outnames(func.funcname))

            writer.write(funcdef)
            writer.write('\n\n\n')
            writer.write(classdef)
            writer.write('\n\n\n')
            funcnames.append(funcname)
            classnames.append(classname)
        # Write "all operations" tuples
        writer.write('ALL_HCTSA_FUNCS = (\n    %s)' % '    '.join('%s,\n' % f for f in funcnames))
        writer.write('\n\n')
        writer.write('ALL_HCTSA_CLASSES = (\n    %s)' % '    '.join('%s,\n' % f for f in classnames))


###################################
# Fix, install, mex the HCTSA library
###################################

def fix_hctsa():
    """Applies some fixes to the HCTSA codebase so they work for us.
    N.B. running this function twice should not lead to any problem...
    """
    # Functions that do not correspond to their file name
    for mfile, wrong_funcname in {'SB_MotifThree.m': 'ST_MotifThree'}.iteritems():
        rename_matlab_func(op.join(HCTSA_DIR, 'Operations', mfile), wrong_funcname)


def mex_hctsa(engine=None):
    """Compiles the mex extensions using the specified engine."""
    # Fix matrix imports
    #
    # In recent versions of matlab, when building mex extensions it is not necessary anymore to import "matrix.h".
    # (importing "mex.h" already informs of all the necessary declarations).
    # Octave mex / mkoctfile goes one step further and refuses to build files importing "matrix.h".
    #
    # Here we just detect such cases and comment the line including matrix.h in HCTSA.
    FILES_INCLUDING_MATRIX = (
        'OpenTSTOOL/mex-dev/Utils/mixembed.cpp',
        'gpml/util/lbfgsb/arrayofmatrices.cpp',
        'gpml/util/lbfgsb/lbfgsb.cpp',
        'Max_Little/steps_bumps_toolkit/ML_kvsteps_core.c',
        'Max_Little/steps_bumps_toolkit/ML_kvsteps_core.cpp',
        'Max_Little/fastdfa/ML_fastdfa_core.c'
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
    # At the moment only OpenTSTool fails to compile under Linux64
    # We will need to modify OpenTSTOOL/mex-dev/makemex.m
    engine.run_text('cd %s' % HCTSA_TOOLBOXES_DIR)
    engine.run_text('compile_mex')


def prepare_engine_for_hctsa(engine):
    """Loads HCTSA and octave dependencies in the engine."""
    # Adds HCTSA to the engine path, so it can be used
    engine.add_path(HCTSA_DIR)
    # Load dependencies from octave-forge
    # See also notes on optional package loading:
    # https://wiki.archlinux.org/index.php/Octave#Using_Octave.27s_installer
    if isinstance(engine, Oct2PyEngine):
        engine.run_text('pkg load signal')
        engine.run_text('pkg load statistics')
        engine.run_text('pkg load parallel')
        engine.run_text('pkg load optim')
        engine.run_text('pkg load econometrics')


def install_hctsa(engine='octave'):
    """Fixes problems with the HCTSA codebase and mexes extensions.

    Parameters
    ----------
    engine: string or MatlabEngine
        The engine to use to build the the mex files
        if 'octave', Oct2PyEngine will be used; if 'matlab', PyMatlabEngine will be used;
        else it must be a MatlabEngine
    """
    if engine is None or engine == 'octave':
        engine = Oct2PyEngine()
    elif engine == 'matlab':
        engine = PyMatlabEngine()
    # Fix some problems with the codebase
    fix_hctsa()
    # Add HCTSA to the engine path
    prepare_engine_for_hctsa(engine)
    # Build the mex files
    mex_hctsa(engine)


###################################
# Entry point
###################################

if __name__ == '__main__':
    import argh
    parser = argh.ArghParser()
    parser.add_commands([install_hctsa, gen_python_bindings, hctsa_summary])
    parser.dispatch()

#
# TODO: batch processing:
#       allow to generate many call lines to group the operations,
#       run them in data already in matlab-land, so we get rid of the excessive call overhead
#       probably use one of (rowfun, colfun, cellfun) in matlab land, put in matlab_utils
#       Interesting :
#          https://github.com/adambard/functools-for-matlab
#          (lambdas are slow in matlab)
#       cellfun is slow in matlab, usually much slower than pathetically slow loops
#          http://www.mathworks.com/matlabcentral/newsreader/view_thread/253815
#          http://stackoverflow.com/questions/18284027/cellfun-versus-simple-matlab-loop-performance
#
# TODO: Configurable
#
# TODO: estimate speed / complexity (hard because it can also depend on outputs)
#
# TODO: create tests with outputs from matlab
#
# TODO: FeatureCategory to MetaFeature (or MetaOps, as Ben seems to call them)
#