# coding=utf-8
"""Catalog for the operations in HCTSA.

HCTSA is a big library of "Operations" together with some extra code to create a database of time-series features
Each "Operation" maps a time-series to some outputs, and each output ("Feature") is a scalar.
An "Operation" represents a call to an "Operator" with certain parameters, producing one or more "Features".
"""
from glob import glob
from itertools import chain
import os.path as op

from pyopy.hctsa.hctsa_config import HCTSA_OPS_FILE, HCTSA_MOPS_FILE, HCTSA_OPERATIONS_DIR, HCTSA_OPS_REDUCED_FILE
from pyopy.code import matlab_funcname_from_filename, parse_matlab_params, parse_matlab_funcdef


class HCTSAFunction(object):
    """A main function (aka "operator") in an mfile, parsed to its main components, from the HCTSA library.

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

    operations : list-like of HCTSAOperation objects
      The different fixed parameters used in the HCTSA code base to call this function.
    """
    def __init__(self,
                 mfile,
                 outstring,
                 params,
                 doc,
                 code,
                 operations):
        super(HCTSAFunction, self).__init__()
        self.mfile = op.abspath(mfile)
        self.funcname = matlab_funcname_from_filename(self.mfile)
        self.outstring = outstring
        self.params = params
        self.doc = doc
        self.code = code
        self.operations = operations

    def add_operation(self, operation):
        """Adds a new operation (call to this function with specific parameter values) to this function."""
        self.operations.append(operation)

    def tags(self):
        """Returns a sorted list with the tags on all operations which arise from this function."""
        return sorted(set(chain.from_iterable(operation.tags() for operation in self.operations)))

    def known_outputs(self):
        """Returns a sorted tuple with the outputs declared in HCTSA metadata for this function."""
        return tuple(sorted(set(operation.known_outputs() for operation in self.operations)))


class HCTSAFeature(object):
    """
    A feature in HCTSA parlance is a single scalar generated from
    the application of an Operator (function) to a time-series
    using concrete parameter values.

    Parameters
    ----------
    featname: string
      The feature name in the HCTSA database (e.g. "CO_HistogramAMI_1_std1_2.min")

    operation: HCTSAOperation
      The operation used to compute this feature (holds function and parameter values information)

    outname: string
      The name of the field in the matlab struct that is outputted by the matlab function
      or None if the function returns already a single scalar (e.g. "min")

    tags: string list
      The tags given to the feature in HCTSA, useful for categorization
    """
    def __init__(self,
                 featname,
                 operation,
                 outname,
                 tags):
        super(HCTSAFeature, self).__init__()
        self.featname = featname
        self.operation = operation
        self.outname = outname
        self.tags = tags


class HCTSAOperation(object):
    """
    An Operation in HCTSA parlance is the combination of
    and Operator (function) and concrete parameter values.

    Parameters
    ----------
    opname : string
      The operation name (e.g. "CO_CompareMinAMI_even_2_80")

    opcall : string
      The operation call string in matlab land (e.g. "CO_CompareMinAMI(y,'even',[2:80])")

    funcname : string
      The function name associated to this operation (e.g. "CO_CompareMinAMI")

    param_values : list
      A list with the values for the function that define this operation (as objects in python land)

    is_commented : bool
      Whether the operation is commented in the HCTSA code

    function : HCTSAFunction
      The function used by this operation

    standardize : bool, default False
      Many operations require the input time series to be standardized (marked by "y" instead of "x" as input)
    """
    def __init__(self,
                 opname,
                 opcall,
                 funcname,
                 param_values,
                 is_commented,
                 function=None,
                 standardize=False):
        super(HCTSAOperation, self).__init__()
        self.opname = opname
        self.opcall = opcall
        self.funcname = funcname
        self.param_values = param_values
        self.is_commented = is_commented
        self.function = function
        self.features = []
        self.standardize = standardize

    def add_feature(self, hctsa_feature):
        """Adds a feature to this operation "known features".

        Parameters
        ----------
        hctsa_feature : HCTSAFeature
            A feature to add to this operation (note that at the moment we allow duplicates)
        """
        self.features.append(hctsa_feature)

    def known_outputs(self):
        """Returns the names of the known outputs for this operation as a tuple."""
        return tuple(sorted(feat.outname for feat in self.features))

    def tags(self):
        """Returns a sorted list with the tags on all features which arise from this operation."""
        return sorted(set(chain.from_iterable(feat.tags for feat in self.features)))

    def has_tag(self, tag):
        return tag in set(chain.from_iterable(feat.tags for feat in self.features))


class HCTSACatalog(object):
    """
    Puts together the information of functions, parameter values and outputs from the HCTSA library.

    Parameters
    ----------
    mfiles_dir: path
      The directory where the matlab m files for the HCTSA operations reside.

    mops_file: path
      The path to the file where we find the map {operation -> (function, param_values)}

    ops_file: path
      The path to the file where we find the map {feature_name -> (operation, output_name)}

    Useful Members
    --------------
    functions_dict : dictionary
      A map {function_name -> HCTSAFunction}

    operations_dict : dictionary
      A map {operation_name -> HCTSAOperation}

    features_dict : dictionary
      A map {feature_name -> HCTSAFeature}
    """

    _catalog = None

    def __init__(self,
                 mfiles_dir=HCTSA_OPERATIONS_DIR,
                 mops_file=HCTSA_MOPS_FILE,
                 ops_file=HCTSA_OPS_FILE,
                 reduced_ops_file=HCTSA_OPS_REDUCED_FILE):
        super(HCTSACatalog, self).__init__()

        self.mfiles_dir = mfiles_dir
        self.mops_file = mops_file
        self.ops_file = ops_file
        self.reduced_ops_file = reduced_ops_file

        self.functions_dict = None
        self.operations_dict = None
        self.features_dict = None
        self.reduced_ops = None

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
        # 1) Get all the operations defined in the MOPS file
        #
        # An operation is a concrete call to an operator with concrete parameters, and can be "commented"
        # (meaning is ignored by the HCTSA code).
        #
        # Each call can return several many named features after the same computation,
        #

        def parse_mops_line(line):
            """Parses a line of the MOPS file.

            An example line:
              'CO_CompareMinAMI(y,'even',[2:80])	CO_CompareMinAMI_even_2_80'

            Gets splitted as:
              operation = 'CO_CompareMinAMI_even_2_80'
              opcall = "CO_CompareMinAMI(y,'even',[2:80])"
              funcname = 'CO_CompareMinAMI'
              params = ['even', '2:80']
              is_commented = False
              is_standardized = True  # Because ts is y
            """
            line, is_commented = manage_comments(line)
            if line is None:  # Ignore commented lines
                return (None,) * 6
            callspec, operation = line.split()
            funcname, _, params = callspec.partition('(')
            funcname = funcname.strip()
            params = params.rpartition(')')[0].strip()
            series, _, params = params.partition(',')  # The first parameter is always the time series
            params = parse_matlab_params(params)
            is_standardized = series == 'y'
            return operation, callspec, funcname, params, is_commented, is_standardized

        self.operations_dict = {}
        try:
            with open(self.mops_file) as reader:
                for line in reader:
                    if not line.strip():
                        continue
                    operationname, callspec, funcname, params, is_commented, is_standardized = parse_mops_line(line)
                    if operationname is None:
                        continue  # Ignore commented lines
                    if operationname in self.operations_dict:
                        raise Exception('Repeated operation: %s' % operationname)
                    self.operations_dict[operationname] = \
                        HCTSAOperation(operationname, callspec, funcname, params, is_commented, standardize=is_standardized)
        except IOError as ex:
            ex.message = 'Cannot find the HCTSA mops file "%s". Maybe run pyopy/hctsa/hctsa_install.py?\n%s' % (
                self.mops_file, ex.message)
            raise

        #
        # 2) Get all the features defined in the OPS file
        # A feature maps an operation to the concrete scalar that needs to be taken out of the function call.
        #

        def parse_ops_line(line):
            """Parses a line of the OPS file.

            An example line:
              CO_CompareMinAMI_even_2_80.min	CO_CompareMinAMI_even_2_80_min	correlation,AMI

            Gets splitted as:
              operation = 'CO_CompareMinAMI_even_2_80'
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
                operationname, outname = outname.split('.')  # Operator returns a struct
            else:
                operationname, outname = outname, None  # Operator returns a scalar
            tags = tags.split(',') if tags is not None else None
            return operationname, outname, featname, tags

        self.features_dict = {}
        with open(self.ops_file) as reader:
            for line in reader:
                if not len(line.strip()):
                    continue
                operationname, outname, featname, tags = parse_ops_line(line)
                if operationname is None:
                    continue  # Ignore commented lines
                operation = self.operations_dict.get(operationname,
                                                     HCTSAOperation(operationname, None, None, None, None))
                feature = HCTSAFeature(featname, operation, outname, tags)
                if featname in self.features_dict:
                    msg = 'Warning: the feature %s is defined more than once, ignoring...' % featname
                    print msg
                    continue  # But actually keep these that are not commented
                    # raise Exception(msg)
                self.features_dict[featname] = feature
                operation.add_feature(feature)  # Other way round

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
            if '% -------------' not in doc:
                # New HCTSA doc style after the function definition, as we talked with Ben
                doc, _, code = code.partition('% -------------')
                code = code.partition('% -----------------------------------'
                                      '-------------------------------------------')[2].partition
            else:
                doc = doc.split('% -------------------------------------')[2]  # Remove header and license
            parameters = parameters[1:]  # The first parameter is always the time series
            operations = [operation for operation in self.operations_dict.values() if operation.funcname == funcname]
            self.functions_dict[funcname] = HCTSAFunction(mfile, outstring, parameters, doc, code, operations)

        #
        # 4) Read the reduced-operations file
        #    These showed worthiness in a feature selection paper
        #
        # FIXME: this is wrongly parsed, mainly because the file is poorly formatted
        operations = set()
        with open(self.reduced_ops_file) as reader:
            for line in (line for line in reader.readlines() if not line.strip().startswith('#') and len(line.strip())):
                operations.add(line.strip().split()[0].split('.')[0])
            self.reduced_ops = tuple(sorted(operations))

    def summary(self):
        report = [
            'Number of operators (functions in mfiles):    %d' % len(self.functions_dict),

            'Number of operations (function + parameters): %d' % len(self.operations_dict),

            'Number of features (operation + outvalue):     %d' % len(self.features_dict),

            'Functions without operation: %s' % sorted(
                set(self.functions_dict.keys()) - set(oper.funcname for oper in self.operations_dict.values())),

            'Operations without functions: %s' % sorted(
                oper.opname for oper in self.operations_dict.values() if
                oper.funcname not in self.functions_dict),
            '    (these are probably calls into other toolboxes)',

            'Features without operations: %s' % sorted(
                fea.featname for fea in self.features_dict.values() if
                fea.operation.opname not in self.operations_dict),
            '    (usually this should be empty)',
        ]
        return '\n'.join(report)

    def default_parameters(self, funcname):
        """Returns default parameter values for a function as one (random)
        of the calls to generate features using such function.
        """
        for operation in self.operations_dict.values():
            if operation.funcname == funcname:
                return operation.param_values
        return []

    def function_parameter_values(self, funcname):
        """Returns all the values that a function takes to span a HTCSA operation."""
        return [oper.param_values for oper in self.functions_dict[funcname].operations()]

    def function_outnames(self, funcname):
        """Returns all the outnames used in HCTSA to extract scalars
        from the application of this function to a timeseries.

        Note that [None] and [] have different meanings here:
          - [None] means the operation output is used as is in HCTSA
          - [] means the operation is not used at all in HCTSA
        """
        outnames = set()
        for operation in self.operations_dict.itervalues():
            if operation.funcname == funcname:
                for feature in operation.features:
                    outnames.add(feature.outname)
        return sorted(outnames)

    def operation(self, opname):
        """:rtype: HCTSAOperation"""
        return self.operations_dict.get(opname, None)

    @staticmethod
    def catalog():
        """:rtype: HCTSACatalog"""
        if HCTSACatalog._catalog is None:
            HCTSACatalog._catalog = HCTSACatalog()
        return HCTSACatalog._catalog

    _whatami2hctsa = None
    _allops = None

    @staticmethod
    def allops():
        if HCTSACatalog._allops is None:
            from hctsa_bindings import HCTSAOperations
            HCTSACatalog._allops = sorted((name, comp[2]) for name, comp in HCTSAOperations.__dict__.iteritems()
                                          if not name.startswith('_'))
        return HCTSACatalog._allops

    @staticmethod
    def what2op(whatid):
        """Finds the correspondence between a whatami id and the HCTSA operator."""
        if HCTSACatalog._whatami2hctsa is None:
            HCTSACatalog._whatami2hctsa = {}
            from hctsa_bindings import HCTSAOperations
            for hctsaop, comp in HCTSACatalog.allops():
                # N.B. this is not unique until we use a Standardizer for these features which require standardisation
                HCTSACatalog._whatami2hctsa[comp.what().id()] = (hctsaop, comp)
        return HCTSACatalog._whatami2hctsa.get(whatid, (None, None))

    @staticmethod
    def must_standardize(operation):
        # Copes with Ben's (x -> normal | y -> standardised) convention
        if isinstance(operation, basestring):
            operation = HCTSACatalog.catalog().operations_dict.get(operation, None)
        return operation is not None and operation.standardize


def summary():
    return HCTSACatalog().summary()

if __name__ == '__main__':
    print summary()
