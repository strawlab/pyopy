# coding=utf-8
"""Python bindings generation for HCTSA."""
from itertools import chain
import inspect

from oscail.common.config import Configurable
from pyopy.hctsa import HCTSA_BINDINGS_FILE, HCTSA_BINDINGS_DIR
from pyopy.hctsa.hctsa_catalog import HCTSACatalog
from pyopy.hctsa.hctsa_data import hctsa_sine
from pyopy.misc import ensure_python_package


class HCTSASuper(Configurable):

    TAGS = ()

    def __init__(self, outnames=None):
        super(HCTSASuper, self).__init__(add_descriptors=False)
        self._outnames = outnames

    def output_names(self, eng=None, x=hctsa_sine()[:40], force=False):
        """
        Returns
        -------
        A string list of the outputs in HCTSA order
        """
        if force or self._outnames is None:
            x = eng.put('outputs_for_ts_87010877aawer98', x)
            out = self.eval(eng, x)
            if isinstance(out, dict):
                self._outnames = sorted(out.keys())  # No easy way to get here the order in the struct
            else:
                self._outnames = None
        return self._outnames

    def has_tag(self, tag='shit'):
        """Returns True iff this operation has the specified tag."""
        return tag in self.TAGS


def gen_python_bindings(hctsa_catalog=None, write_function_too=False):
    """Generates the python bindings to the library operators."""

    def gen_function(funcname, parameters, doc, defaults=None, prefix='HCTSA'):
        """Generates python bindings for calling the feature extractors."""

        pyfuncname = '%s_%s' % (prefix, funcname)

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

    def gen_class_from_function(hctsa_function, function_prefix='HCTSA_'):

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
            indent1 + 'def __init__(self, %s):' % args_string,
            constructor_string,
            eval_method,
        ]
        return mlab_name, '\n'.join(code)

    def gen_class_from_function_string(hctsa_function, func_params, function_prefix='HCTSA_', catalog=None):

        if catalog is None:
            catalog = HCTSACatalog()

        # Our indentation levels
        indent1 = ' ' * 4
        indent2 = ' ' * 8

        # Avoid shadowing + work well with oscail
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
        if len(func_params) == 0:
            eval_method = indent1 + '@staticmethod\n'
            eval_method += indent1 + 'def eval(eng, x):\n'
            eval_method += indent2 + body
        else:
            eval_method = indent1 + 'def eval(self, eng, x):\n'
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

    def gen_categories_class(catalog=None, add_commented_out=False):
        """Returns text with all the metaops in the current HCTSA release under a class namespace."""
        if catalog is None:
            catalog = HCTSACatalog()
        lines = []
        for fname, func in sorted(catalog.functions_dict.items()):
            for category in func.categories:
                if not add_commented_out and category.is_commented:
                    continue
                # generate the call string
                # avoid shadowing buitins with parameters...
                params = [{'ord': 'ordd'}.get(param, param) for param in func.params]
                values = category.param_values
                params_string = '' if not values else \
                    ', '.join('%s=%r' % (name, value) for name, value in zip(params, values))

                def chunks(l, n):
                    for i in xrange(0, len(l), n):
                        yield l[i:i+n]
                for outs in chunks(category.known_outputs(), 5):
                    lines.append('# outs: %s' %
                                 ','.join(map(str, outs) if category.known_outputs() else ''))
                lines.append('# tags: %s' %
                             ','.join(map(str, category.tags()) if category.tags() else ''))
                lines.append('%s =\\' % category.catname)
                instline = '    %s(%s)\n' % (fname, params_string)
                if len(instline) < 110:
                    lines.append(instline)
                else:
                    first_comma = instline.find(',', 90)
                    lines.append(instline[:first_comma + 1])
                    lines.append(' ' * len('    %s(' % fname) + instline[first_comma+2:])
        lines.append('@staticmethod')
        lines.append('def all():')
        lines.append('    return sorted((name, comp) for name, comp in HCTSA_Categories.__dict__.iteritems()')
        lines.append('                  if not name.startswith(\'_\') and not name == \'all\')')
        lines = ['    %s' % line for line in lines]
        return 'class HCTSA_Categories(object):\n' \
               '    """Namespace for HCTSA selected features."""' \
               '\n\n%s' % '\n'.join(lines)

    # Read-in the catalog
    if hctsa_catalog is None:
        hctsa_catalog = HCTSACatalog()
    # Ensure that the destination python package exists
    ensure_python_package(HCTSA_BINDINGS_DIR)
    with open(HCTSA_BINDINGS_FILE, 'w') as writer:
        # Bindings imports
        binding_imports = (
            'from pyopy.matlab_utils import MatlabSequence',
            'from pyopy.hctsa.hctsa_bindings_gen import HCTSASuper')
        exec '\n'.join(binding_imports) in globals()  # We are using nasty execs around that need these imports
        # Write the header
        writer.write('# coding=utf-8\n')
        writer.write('\n'.join(binding_imports) + '\n\n\n')
        # Write the functions and classes
        funcnames = []
        classnames = []
        exclusions = {'PP_PreProcess', }  # PP_PreProcess oes not extract features
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
                exec funcdef in globals()  # Nasty, but avoids second passes and acts as a sanity check
                classname, classdef = gen_class_from_function(eval(funcname))
            else:
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
        writer.write(gen_categories_class())

if __name__ == '__main__':
    gen_python_bindings()