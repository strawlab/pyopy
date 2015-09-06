# coding=utf-8
"""Python bindings generation for HCTSA."""
from itertools import chain

from whatami import whatable

from pyopy.hctsa.hctsa_config import HCTSA_BINDINGS_FILE, HCTSA_BINDINGS_DIR
from pyopy.hctsa.hctsa_catalog import HCTSACatalog
from pyopy.hctsa.hctsa_data import hctsa_sine
from pyopy.base import PyopyEngines
from pyopy.hctsa.hctsa_transformers import hctsa_prepare_input
from pyopy.misc import ensure_python_package


@whatable
class HCTSASuper(object):

    TAGS = ()

    def __init__(self, outnames=None, eng=None):
        self._outnames = outnames
        self._eng = eng

    def _infer_eng(self, eng):
        if eng is None:
            eng = self._eng if self._eng is not None else PyopyEngines.default()
        return eng

    def output_names(self, eng=None, x=None, force=False):
        """
        Returns
        -------
        A string list of the outputs in HCTSA order
        """
        eng = self._infer_eng(eng)
        if x is None:
            x = hctsa_sine()[:40]
        if force or self._outnames is None:
            # N.B. this is not deterministic inference, as sometimes the number of outputs depend on the T.S.
            # Is there an "upper bound on the amount of outputs"?
            x = eng.put('outputs_for_ts_87010877aawer98', x)  # FIXME: put this ts in the engine on prepare
            out = self.transform(x, eng=eng)
            if isinstance(out, dict):
                self._outnames = sorted(out.keys())  # No easy way to get here the order in the struct
            else:
                self._outnames = None
        return self._outnames

    def transform(self, x, eng=None):
        eng = self._infer_eng(eng)
        return self._eval_hook(eng, x)

    def compute(self, x, eng=None):
        return self.transform(x, eng=eng)

    def _eval_hook(self, eng, x):
        raise NotImplementedError()

    def has_tag(self, tag='shit'):
        """Returns True iff this operation has the specified tag."""
        return tag in self.TAGS

    def use_eng(self, eng):
        self._eng = eng


@whatable(force_flag_as_whatami=True)
class HCTSAOperation(object):
    def __init__(self, operation_name, matlab_call, operation):
        super(HCTSAOperation, self).__init__()
        self.name = operation_name
        self.matlab_call = matlab_call
        self.operation = operation
        self._must_standardize = None

    def what(self):
        return self.operation.what()

    def must_standardize(self):
        if self._must_standardize is None:
            from hctsa_catalog import HCTSACatalog
            self._must_standardize = HCTSACatalog.must_standardize(self.name)
        return self._must_standardize

    def compute(self, x, y=None, eng=None):
        if self.must_standardize() and y is None:
            x = hctsa_prepare_input(x, z_scored=True) if y is None else y
        else:
            x = hctsa_prepare_input(x, z_scored=False)
        return self.operation.compute(x, eng=eng)

    def __call__(self, x, y=None, eng=None):
        return self.compute(x, y=y, eng=eng)

    def _as_tuple(self):
        return self.name, self.matlab_call, self.operation

    def __getitem__(self, item):
        return self._as_tuple()[item]

    def __iter__(self):
        return iter(self._as_tuple())

    def __repr__(self):
        return self.__class__.__name__ + self._as_tuple().__repr__()


def gen_bindings(hctsa_catalog=None, write_function_too=False):
    """Generates the python bindings to the library operators."""

    def gen_function(funcname, parameters, doc, defaults=None, prefix='HCTSA', varargin_as_args=False):
        """Generates python bindings for calling the feature extractors."""

        pyfuncname = '%s_%s' % (prefix, funcname)

        # varargin support (http://www.mathworks.com/help/matlab/ref/varargin.html)
        if len(parameters) > 0 and parameters[-1] == 'varargin' and varargin_as_args:
            parameters[-1] = '*args'
            raise NotImplementedError()
        # this would require to support default values longer than the number of parameters
        # this would need further work on calling to support pure numeric args via the cell-trick

        # avoid shadowing buitins with parameters...
        parameters = [{'ord': 'ord_'}.get(param, param)for param in parameters]
        # build the function signature
        if defaults is None:
            defaults = [None] * len(parameters)
        else:
            defaults.extend([None] * (len(parameters) - len(defaults)))
        defaults = map(lambda x: float(x) if isinstance(x, int) else x, defaults)  # all double
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

        # function body
        body = []
        for i, parameter in enumerate(parameters):
            body.append('    %s %s is None:' % ('if' if i == 0 else 'elif', parameter))
            body.append('        return eng.run_function(1, \'%s\', x, %s)' % (funcname, ', '.join(parameters[0:i])))
        body.append('    return eng.run_function(1, \'%s\', x, %s)' % (funcname, ', '.join(parameters)))
        body_line = '\n'.join(body)

        # assemble all together
        return pyfuncname, '\n'.join([defline, doc_line, body_line])

    def gen_class_from_function_string(hctsa_function, func_params, function_prefix='HCTSA_', catalog=None):

        if catalog is None:
            catalog = HCTSACatalog.catalog()

        # Our indentation levels
        indent1 = ' ' * 4
        indent2 = ' ' * 8

        # Avoid shadowing + work well with whatami
        func_params = [{'ord': 'ordd'}.get(param, param) for param in func_params]
        hctsa_function = hctsa_function.replace('ord_', 'ordd')

        # Split the generated function code...
        # All this nastiness out of lazyness (did first function generation and do not feel like redoing right now)
        deflines, _, body = hctsa_function.partition('):\n')
        name = deflines[4:].partition('(')[0][len(function_prefix):]
        args_string = deflines.partition('(eng, x, ')[2]
        docstring, _, body = body.rpartition('"""')
        docstring += '"""'

        # parameter read -> class member read
        for param in func_params:
            body = body.replace('if %s is' % param, 'if self.%s is' % param)
            if 'varargin' == param:

                # dirty hack to force varagint to be transferred as cells
                # can fail in many instances (e.g. tuples should be passed as ((1,2),) instead of just (1,2))
                # works for HCTSA as it is at the moment, but varargin is not generally supported
                # we need to fully implement it by translating to python *args
                body = body.replace(
                    ', varargin',
                    ", self.varargin + ('_celltrick_',) if isinstance(self.varargin, tuple) "
                    "else (self.varargin, '_celltrick_')")
            else:
                body = body.replace(', %s' % param, ', self.%s' % param)

        body = body.strip()

        # known outputs and tags
        outputs_string = indent1 + 'KNOWN_OUTPUTS_SIZES = %r' % (
            tuple(len(k) for k in catalog.functions_dict[name].known_outputs()),)
        tags_string = indent1 + 'TAGS = %r' % (tuple(catalog.functions_dict[name].tags()), )
        if len(tags_string) > 120:
            comma = tags_string.find(', ', 90)
            tags_string = tags_string[:comma] + '\n' + ' ' * len(indent1 + 'TAGS = (') + tags_string[comma+2:]

        # Constructor body
        if args_string:
            lines = args_string.splitlines()
            if len(lines) == 2:  # cosmetics
                constructor_string = indent1 + 'def __init__(self, %s\n%s' % (
                    lines[0], ' ' * len('    def __init__(') + lines[1].strip() + '):\n')
            else:
                constructor_string = indent1 + 'def __init__(self, %s):\n' % args_string
        else:
            constructor_string = indent1 + 'def __init__(self):\n'
        constructor_string += indent2 + 'super(%s, self).__init__()\n' % name
        if len(func_params) > 0:
            constructor_string += indent2 + indent2.join('self.%s = %s\n' % (param, param) for param in func_params)

        # eval method
        eval_method = indent1 + 'def _eval_hook(self, eng, x):\n'
        eval_method += indent1 + '\n'.join(indent1 + line for line in body.splitlines())

        # cosmetics
        def break_long(line):
            if len(line) < 120:
                return [line]
            comma = line.find(', ', 90)
            return [line[:comma+1],
                    ' ' * len(line.partition('run_function(')[0] + 'run_function(') + line[comma + 2:]]
        eval_method = '\n'.join(chain.from_iterable(map(break_long, eval_method.splitlines())))

        # put all together
        code = [
            'class %s(HCTSASuper):' % name,
            docstring + '\n',
            outputs_string + '\n',
            tags_string + '\n',
            constructor_string,
            eval_method,
        ]
        return name, '\n'.join(code)

    def gen_operations_class(catalog=None, add_commented_out=False):
        """Returns text with all the metaops in the current HCTSA release under a class namespace."""
        if catalog is None:
            catalog = HCTSACatalog()
        lines = []
        for fname, func in sorted(catalog.functions_dict.items()):
            for operation in func.operations:
                if not add_commented_out and operation.is_commented:
                    continue
                # generate the call string
                # avoid shadowing buitins with parameters...
                params = [{'ord': 'ordd'}.get(param, param) for param in func.params]
                values = operation.param_values
                params_string = '' if not values else \
                    ', '.join('%s=%r' % (name, value) for name, value in zip(params, values))

                def chunks(l, n):
                    for i in range(0, len(l), n):
                        yield l[i:i+n]
                for outs in chunks(operation.known_outputs(), 5):
                    lines.append('# outs: %s' %
                                 ','.join(map(str, outs) if operation.known_outputs() else ''))
                lines.append('# tags: %s' %
                             ','.join(map(str, operation.tags()) if operation.tags() else ''))
                lines.append('%s = HCTSAOperation(' % operation.opname)
                lines.append('    \'%s\',' % operation.opname)
                lines.append('    %r,' % operation.opcall)
                instline = '    %s(%s))\n' % (fname, params_string)
                if len(instline) < 110:
                    lines.append(instline)
                else:
                    first_comma = instline.find(',', 90)
                    lines.append(instline[:first_comma + 1])
                    lines.append(' ' * len('    %s(' % fname) + instline[first_comma+2:])

        lines = ['    %s' % line for line in lines]
        return 'class HCTSAOperations(object):\n' \
               '    """Namespace for HCTSA selected operations."""' \
               '\n\n%s' % '\n'.join(lines)

    # Read-in the catalog
    if hctsa_catalog is None:
        hctsa_catalog = HCTSACatalog()
    # Ensure that the destination python package exists
    ensure_python_package(HCTSA_BINDINGS_DIR)
    with open(HCTSA_BINDINGS_FILE, 'w') as writer:
        # Bindings imports
        binding_imports = (
            'from pyopy.base import MatlabSequence',
            'from pyopy.hctsa.hctsa_bindings_gen import HCTSASuper, HCTSAOperation')
        exec '\n'.join(binding_imports) in globals()  # We are using nasty execs around that need these imports
        # Write the header
        writer.write('# coding=utf-8\n')
        writer.write('\n'.join(binding_imports) + '\n\n\n')
        # Write the functions and classes
        funcnames = []
        classnames = []
        exclusions = {'PP_PreProcess', }  # PP_PreProcess does not extract features
        for func in sorted(hctsa_catalog.functions_dict.values(), key=lambda f: f.funcname):
            if func.funcname in exclusions:
                continue
            funcname, funcdef = gen_function(func.funcname,
                                             func.params,
                                             func.doc,
                                             hctsa_catalog.default_parameters(func.funcname))
            if write_function_too:
                writer.write(funcdef)
                writer.write('\n\n\n')

            classname, classdef = gen_class_from_function_string(funcdef, func.params, catalog=hctsa_catalog)

            writer.write(classdef)
            writer.write('\n\n\n')
            funcnames.append(funcname)
            classnames.append(classname)
        # Write "all operations" tuples
        if write_function_too:
            writer.write('HCTSA_ALL_FUNCS = (\n    %s)' % '    '.join('%s,\n' % f for f in funcnames))
            writer.write('\n\n')
        writer.write('HCTSA_ALL_CLASSES = (\n    %s)' % '    '.join('%s,\n' % f for f in classnames))
        writer.write('\n\n\n')
        writer.write(gen_operations_class())

if __name__ == '__main__':
    gen_bindings()
