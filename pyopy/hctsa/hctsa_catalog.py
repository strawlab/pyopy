# coding=utf-8
"""Catalog for the operations in HCTSA.

HCTSA is a big library of "Operations" together with some extra code to create a database of time-series features
Each "Operation" maps a time-series to some outputs, and each output ("Feature") is a scalar.
A "Category" represents a call to an "Operator" with certain parameters, producing one or more "Features".
"""
from glob import glob
from itertools import chain
import os.path as op
from pyopy.hctsa import HCTSA_OPS_FILE, HCTSA_MOPS_FILE, HCTSA_OPERATIONS_DIR
from pyopy.matlab_utils import matlab_funcname_from_filename, parse_matlab_params, parse_matlab_funcdef


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

    categories : list-like of HCTSACategory objects
        The different fixed parameters used in the HCTSA code base to call this function.
    """
    def __init__(self,
                 mfile,
                 outstring,
                 params,
                 doc,
                 code,
                 categories):
        super(HCTSAFunction, self).__init__()
        self.mfile = op.abspath(mfile)
        self.funcname = matlab_funcname_from_filename(self.mfile)
        self.outstring = outstring
        self.params = params
        self.doc = doc
        self.code = code
        self.categories = categories

    def add_category(self, category):
        """Adds a new category (call to this function with specific parameter values) to this function."""
        self.categories.append(category)

    def tags(self):
        """Returns a sorted list with the tags on all categories which arise from this function."""
        return sorted(set(chain.from_iterable(cat.tags() for cat in self.categories)))

    def known_outputs(self):
        """Returns a sorted tuple with the outputs declared in HCTSA metadata for this function."""
        return tuple(sorted(set(cat.known_outputs() for cat in self.categories)))


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

    function : HCTSAFunction
        The function used to generate this category of features
    """
    def __init__(self,
                 catname,
                 funcname,
                 param_values,
                 is_commented,
                 function=None):
        super(HCTSAFeatureCategory, self).__init__()
        self.catname = catname
        self.funcname = funcname
        self.param_values = param_values
        self.is_commented = is_commented
        self.function = function
        self.features = []

    def add_feature(self, hctsa_feature):
        """Adds a feature to this category "known features".

        Parameters
        ----------
        hctsa_feature : HCTSAFeature
            A feature to add to this category (note that at the moment we allow duplicates)
        """
        self.features.append(hctsa_feature)

    def known_outputs(self):
        """Returns the names of the known outputs for this category as a tuple."""
        return tuple(sorted(feat.outname for feat in self.features))

    def tags(self):
        """Returns a sorted list with the tags on all features which arise from this category."""
        return sorted(set(chain.from_iterable(feat.tags for feat in self.features)))

    def has_tag(self, tag):
        return tag in set(chain.from_iterable(feat.tags for feat in self.features))


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
        A map {function_name -> HCTSAFunction}

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
            categories = [cat for cat in self.categories_dict.values() if cat.funcname == funcname]
            self.functions_dict[funcname] = HCTSAFunction(mfile, outstring, parameters, doc, code, categories)

        #
        # 4) Make easy the link:
        #    function -> {category -> {features}}
        #

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
        return [cat.param_values for cat in self.functions_dict[funcname].categories()]

    def function_outnames(self, funcname):
        """Returns all the outnames used in HCTSA to extract scalars
        from the application of this function to a timeseries.

        Note that [None] and [] have different meanings here:
          - [None] means the operation output is used as is in HCTSA
          - [] means the operation is not used at all in HCTSA
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

#
# TODO: FeatureCategory to MetaFeature (or MetaOps, as Ben seems to call them)#