# This file is a part of OMPC (http://ompc.juricap.com/)

import sys
sys.path += ['../outside/ply']
OCTAVE = False

# TODO
# 
# - make the no ';' print-out an option, the output is ugly
# - add 1 2, or anything like that, everything after the NAME is considered 
#   a string
# - print -png ... and similar

_keywords = ["break", "case", "catch", "continue", "else", "elseif", "end", 
             "for", "function", "global", "if", "otherwise", "persistent", 
             "return", "switch", "try", "while"]

_octave_keywords = ["endif", "endwhile", "endfunction", "endswicth", "endfor"]

# functions that are known to not return a value, this will make the
# resulting code prettier
_special = ['pause', 'plot', 'hold', 'axis', 'pcolor', 'colorbar',
            'pause', 'disp', 'colormap', 'set', 'title',
            'xlabel', 'ylabel']

_ompc_reserved = ['mfunction', 'mcat', 'mcellarray', 'marray', 'mstring']

def isompcreserved(name):
    return name in _ompc_reserved

reserved = dict( (x.lower(), x.upper()) for x in _keywords )

if OCTAVE:
    reserved.update( dict( (x.lower(), 'END') for x in _octave_keywords ) )

tokens = [
    'NAME', 'NUMBER', 'STRING',
    'COMMA', 'SEMICOLON', 'NEWLINE',
    'DOTTIMES', 'DOTDIVIDE', 'DOTPOWER', 'DOTRDIVIDE',
    'NOTEQUAL', 'ISEQUAL', 'TRANS', 'CONJTRANS',
    'LESS', 'GREATER', 'LESSEQUAL', 'GREATEREQUAL',
    'AND', 'OR', 'NOT', 'ELOR', 'ELAND',
    'LBRACKET', 'RBRACKET', 'LCURLY', 'RCURLY', 'LPAREN', 'RPAREN',
    'LAMBDA',
    'COMMENT',
    ] + reserved.values()

literals = ['=', '+', '-', '*', '/', '\\', '^', ':', "'", '.']

states = (
    ('comment', 'exclusive'),
    ('inlist',  'inclusive'),
    ('inparen', 'inclusive'),
    )

# def t_comment(t):
#     r'%(.*)'
#     t.type = 'COMMENT'
#     t.value = '%s'%t.value[1:]
#     t.lexer.lineno += 1
#     return t

def t_LPAREN(t):
    r'[({]'
    t.lexer.push_state('inparen')
    return t

def t_inparen_END(t):
    'end'
    t.value = 'end'
    t.type = 'NUMBER'
    return t

def t_inparen_RPAREN(t):
    r'[)}]'
    t.lexer.pop_state()
    return t

def t_LBRACKET(t):
    r'\['
    t.lexer.push_state('inlist')
    return t

def t_inlist_RBRACKET(t):
    r'\]'
    t.lexer.pop_state()
    return t

def t_LCURLY(t):
    r'\{'
    t.lexer.push_state('inlist')
    return t

# cannot do this because [a(1,2) b] = min(1:4);
#def t_inlist_COMMA(t):
#    r','
#    t.type = 'LISTCOMMA'
#    return t

def t_inlist_RCURLY(t):
    r'\}'
    t.lexer.pop_state()
    return t

t_COMMA = ','
t_SEMICOLON = r';'

# Comments
def t_PERCENT(t):
    r'%'
    t.lexer.push_state('comment')

def t_comment_body(t):
    r'([^\n]+)'
    t.type = 'COMMENT'
    t.lexer.pop_state()
    return t

t_comment_ignore = '.*'

def t_comment_error(t):
    pass


# Tokens

t_DOTTIMES = r'\.\*'
t_DOTDIVIDE = r'\./'
t_DOTRDIVIDE = r'\.\\'
t_DOTPOWER = r'\.\^'
t_NOTEQUAL = r'~='
t_ISEQUAL = r'=='
t_LESS = r'<'
t_GREATER = r'>'
t_LESSEQUAL = r'<='
t_GREATEREQUAL = r'>='
t_ELAND = r'&'
t_ELOR = '\|'
t_AND = r'&&'
t_OR = '\|\|'
t_NOT = '~'

def t_NAME(t):
    r'[a-zA-Z][a-zA-Z0-9_]*'
    t.type = 'NAME'
    if t.value in reserved:
        t.type = reserved.get(t.value)    # Check for reserved words
    else:
        t.value = _check_name(t.value)
    return t

t_LAMBDA = r'@'

t_TRANS = r"\.'"
t_CONJTRANS = r"'"

def t_STRING(t):
    r"'((?:''|[^\n'])*)'"
    pos = t.lexer.lexpos - len(t.value)
    if pos == 0:
        return t
    
    prec = t.lexer.lexdata[pos-1]
    if prec == '.':
        t.value = ".'"
        t.type = "TRANS"
        t.lexer.lexpos = pos + 2
    elif prec in ' \t[{(=;,\n':
        # it's a string, translate "''" to 
        t.value = "'%s'"%t.value[1:-1].replace("\\", r"\\")
        t.value = "'%s'"%t.value[1:-1].replace("''", r"\'")
    else:
        t.value = "'"
        t.type = "CONJTRANS"
        t.lexer.lexpos = pos + 1
    return t

def t_NUMBER(t):
    r'(?:\d+\.\d*|\d*\.\d+|\d+)(?:[e|E]-?\d+|)'
    try:
        float(t.value)
    except ValueError:
        _print_error("Is this really a float?", t.value)
    return t

def t_COMMENT(t):
    r'%'
    global _comment
    _comment = t.value
    t.lexer.lineno += 1
    #pass
    # No return value. Token discarded

t_ignore = " \t"

def t_NEWLINE(t):
    r'\n'
    pass

# semicolon has a different function inside of [] and {}
def t_inlist_SEMICOLON(t):
    r';'
    t.type = 'COMMA'
    t.value = 'SEMICOLON'
    return t
    #pass

def t_error(t):
    _print_error("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)
    
# Build the lexer
import lex
lex.lex()

# Parsing rules

precedence = (    
    ('left', 'TRANS', 'CONJTRANS'),
    ('nonassoc', 'LESS', 'GREATER'),
    ('left', '+', '-'),
    ('left', '*', '/', 'DOTTIMES', 'DOTDIVIDE'),
    ('left', '^', 'DOTPOWER'),
    ('right', 'UMINUS', 'UPLUS'),
    )

# dictionary of names
names = { }
_key_stack = []
_switch_stack = []
_tabs = 0
_comment = None
TABSHIFT = 4

def _reset():
    global _tabs, names, _key_stack, _switch_stack, _comment, TABSHIFT
    names = { }
    _key_stack = []
    _switch_stack = []
    _tabs = 0
    _comment = None
    TABSHIFT = 4

_reset()

def _gettabs():
    global _tabs, TABSHIFT
    return ' '*_tabs

def _print3000(*args,**kwargs):
    """Emulation of Py3k's print.
    """
    from sys import stdout
    sep = kwargs.get('sep',' ')
    of = kwargs.get('file',stdout)
    end = kwargs.get('end','\n')
    of.write(sep.join(map(str,args)))
    of.write(end)

_outfile = None
def _print(src):
    global _outfile
    ss = src.split('\n')
    for x in ss[:-1]:
        _print3000(' '*_tabs + x, sep='', end='\n', file=_outfile)
    _print3000(' '*_tabs + ss[-1], sep='', end='', file=_outfile)

_errors = []
def _print_error(*args, **kwargs):
    """Error output.
    This function should be used for output of all errors.
    """
    global _errors, _lineno
    from sys import stderr
    sep = kwargs.get('sep',' ')
    of = kwargs.get('file', stderr)
    end = kwargs.get('end','\n')
    if file is None:
        _errors.append(' '.join(args))
    else:
        d = {'sep':sep, 'file':of}
        _print3000(**d)
        _print3000(*args, **d)
        _print3000("On line: %d!"%(_lineno), **d)

        
def _pop_from_key_stack():
    global _key_stack
    if len(_key_stack) < 1:
        _print_error('An "end" without matching keyword!')
        _reset()
        return None
    return _key_stack.pop()
        
def p_statement_list(p):
    '''statement_list : statement
                      | statement COMMA
                      | statement SEMICOLON'''
    if p[1] and p[1][-1] == ';':
        p[0] = _print_statement(p[1][:-1], ';', p[0])
    else:
        p[0] = _print_statement(p[1], len(p) > 2 and p[2] or None, p[0])

_lvalues = []
_knoend = list(_keywords)
_knoend.remove('end')
def _print_statement(x, send, p0):
    global _lvalues, _key_stack, _tabs
    #print >>sys.stderr, '###', x, send, p0
    finish = ''
    if p0 and p0.strip()[-1] not in ':;': finish = '; '
    res = x
    # don't print results of keyword statements and commands, FIXME
    xs = x.strip() and x.strip().split()[0]
    dedent = False
    if not xs:
        pass
    elif xs[0] == '@':
        assert len(_key_stack) == 1 and _key_stack[0] == 'function'
        _pop_from_key_stack()
        _tabs = TABSHIFT
        dedent = True
    #xs in _special or \
    elif xs in _keywords or \
         xs[:2] == '__' or xs in ['elif', 'else:']:
        if xs not in ['end', 'break', 'continue', 'return', 'global']:
            dedent = True
    elif send is None or send == ',':
        # we need to print also the result
        if _lvalues:
            for lv in _lvalues:
                res += '; print %s'%lv
    _lvalues = []
    if dedent: _tabs -=  TABSHIFT
    _print(finish+res)
    if dedent: _tabs +=  TABSHIFT
    return res

def p_statement_list2(p):
    '''statement_list : statement_list SEMICOLON statement
                      | statement_list COMMA statement
                      | statement_list statement'''
    # the statement list has been printed already
    p[0] = _print_statement('\n'+p[-1], p[-1][-1] == ';' and ';' or None, p[0])

def p_statement_finished(p):
    '''statement : statement SEMICOLON
                 | statement COMMA'''
    p[0] = p[1] + (p[2] == ';' and ';' or '')

def p_statement_expr(p):
    '''statement : expression'''
    p[0] = p[1]

def p_statement_function(p):
    '''statement : FUNCTION LBRACKET name_list RBRACKET "=" NAME LPAREN name_list RPAREN
                 | FUNCTION LBRACKET name_list RBRACKET "=" NAME
                 | FUNCTION NAME "=" NAME LPAREN name_list RPAREN
                 | FUNCTION NAME "=" NAME
                 | FUNCTION NAME LPAREN name_list RPAREN
                 | FUNCTION NAME'''
    global _tabs, _key_stack, _func_name
    argout, fname, argin = None, None, None
    if '=' in p:
        if p[2] == '[':
            argout, fname = p[3], p[6]
            if '(' in p: argin = p[8]
        else:
            argout, fname = p[2], p[4]
            if '(' in p: argin = p[6]
    else:
        fname = p[2]
        if '(' in p: argin = p[4]
    # split argin and make all of them equal None
    # if one of the is varargin, change it to *varargin
    argin = [ x.strip() for x in argin.split(',') ]
    last = []
    if 'varargin' in argin:
        if argin[-1] != 'varargin':
            p_error(p)
        argin.pop()
        last = ['*varargin']
    argin = ', '.join([ '%s=None'%x for x in argin ] + last)
    if argout is None:
        argout = ''
    p[0] = '@mfunction("%s")\ndef %s(%s):'%(argout, fname, argin)
    _func_name = fname
    _key_stack.append('function')
    _tabs += TABSHIFT

def p_expression_lambda_handle(p):
    '''expression : LAMBDA NAME'''
    # function handle
    p[0] = p[1]

def p_expression_name_list(p):
    '''name_list : name_list COMMA NAME'''
    p[0] = '%s, %s'%(p[1], p[3])

def p_expression_name_list_2(p):
    '''name_list : NAME'''
    p[0] = p[1]

def p_expression_lambda(p):
    '''expression : LAMBDA LPAREN name_list RPAREN expression'''
    p[0] = 'lambda %s: %s'%(p[3], p[5])

# def '''statement : CLASSDEF NAME'''
#    pass

# properties 
# methods
# events 
    
def p_statement_for(p):
    '''statement : FOR NAME "=" expression'''
    global _tabs, _key_stack
    p[0] = 'for %s in %s:'%(p[2], p[4])
    _key_stack.append('for')
    #_print(p[0])
    _tabs += TABSHIFT

def p_statement_while(p):
    '''statement : WHILE expression'''
    global _tabs, _key_stack
    p[0] = 'while %s:'%p[2]
    _key_stack.append('while')
    #_print(p[0])
    _tabs += TABSHIFT

def p_statement_if(p):
    '''statement : IF expression'''
    global _key_stack, _tabs
    p[0] = 'if %s:'%p[2]
    _key_stack.append('if')
    #_print(p[0])
    _tabs += TABSHIFT

def p_statement_elseif(p):
    '''statement : ELSEIF expression'''
    global _tabs, _key_stack
    # FIXME if p is cellarray we should copare with in
    p[0] = 'elif %s:'%p[2]
    assert _key_stack[-1] == 'if'
    _tabs -= TABSHIFT
    #_print(p[0])
    _tabs += TABSHIFT

def p_statement_else(p):
    '''statement : ELSE'''
    global _tabs, _key_stack
    # FIXME if p is cellarray we should copare with in
    p[0] = 'else:'
    assert _key_stack[-1] == 'if'
    _tabs -= TABSHIFT
    #_print(p[0])
    _tabs += TABSHIFT

def p_statement_break(p):
    """statement : BREAK"""
    p[0] = 'break'

def p_statement_continue(p):
    """statement : CONTINUE"""
    p[0] = 'continue'
    #_print(p[0])

def p_statement_return(p):
    """statement : RETURN"""
    p[0] = 'return'

def p_statement_switch(p):
    '''statement : SWITCH expression'''
    global _tabs, _key_stack, _switch_stack
    svar = '__switch_%d__'%len(_switch_stack)
    p[0] = '%s = %s\nif 0:\n%spass'%(svar, p[2], ' '*TABSHIFT)
    _key_stack.append('switch')
    _switch_stack.append( svar )
    _tabs += TABSHIFT
    #_print(p[0])

def p_statement_case(p):
    '''statement : CASE expression'''
    global _tabs, _key_stack, _switch_stack
    # FIXME if p is cellarray we should copare with in
    p[0] = 'elif %s == %s:'%(_switch_stack[-1], p[2])
    assert _key_stack[-1] == 'switch'
    #_tabs -= TABSHIFT
    #_print(p[0])
    #_tabs += TABSHIFT

def p_statement_otherwise(p):
    """statement : OTHERWISE"""
    global _key_stack
    p[0] = 'else:'
    assert _key_stack[-1] == 'switch'
    #_tabs -= TABSHIFT
    #_print(p[0])
    #_tabs += TABSHIFT

def p_statement_global(p):
    """statement : GLOBAL list_spaces"""
    p[0] = 'global %s'%p[2]
    #_print(p[0])

_func_name = None
def p_statement_persistent(p):
    """statement : PERSISTENT list_spaces"""
    global _func_name
    # FIXME, store in in a module or thread ???
    if _func_name is None:
        _print_error('"persistent" outside of a function block!')
    p[0] = 'global __persistent__\n'
    p[0] += "__persistent__['%s'] = '%s'"%(_func_name, p[2])

def p_expression_list_space(p):
    '''list_spaces : list_spaces NAME'''
    p[0] = '%s, %s'%(p[1], p[2])

def p_expression_list_space_2(p):
    '''list_spaces : NAME'''
    p[0] = p[1]

def p_statement_try(p):
    '''statement : TRY'''
    global _tabs, _key_stack
    p[0] = 'try:'
    _key_stack.append('try')
    _tabs += TABSHIFT

def p_statement_catch(p):
    '''statement : CATCH'''
    global _tabs, _key_stack
    p[0] = 'except:'%(_switch_stack[-1], p[2])
    assert _key_stack[-1] == 'try'

def p_statement_end(p):
    'statement : END'
    global _tabs, _key_stack, _switch_stack
    _tabs -= TABSHIFT
    p[0] = 'end'
    kw = _pop_from_key_stack()
    if kw == 'switch':
        _switch_stack.pop()

def _getname(lname):
    pos = lname.find('(')
    if pos == -1:
        pos = lname.find('{')
    if pos == -1:
        return lname
    return lname[:pos]

def p_statement_assign(p):
    '''statement : name_sub "=" expression
                 | name_attr "=" expression
                 | exprmcat "=" expression
                 | NAME "=" expression'''
    global _lvalues
    lname = p[1]
    if lname[0] == '[':
        # [...]
        ns = []
        for x in lname[1:-1].split(','):
            ln = _getname(x.strip())
            names[ln] = p[3]
            _lvalues += [ln]
    elif '(' in lname:
        p[1] = '%s.lvalue'%lname
        lname = _getname(lname)
        _lvalues = [lname]
        names[lname] = '%s'%p[3]
    else:
        names[lname] = '%s'%p[3]
        _lvalues = [lname]
    p[0] = '%s = %s'%(p[1], p[3])


def p_statement_nogroup(p):
    """statement : NAME NAME
                 | NAME NUMBER"""
    # treating cases like "hold on, axis square"
    p[0] = '%s("%s")'%(p[1], p[2])

def p_expr_list(p):
    '''exprlist : exprlist COMMA expression'''
    p[0] = '%s, %s'%(p[1], p[3])

def p_expr_list_2(p):
    'exprlist : expression'
    p[0] = p[1]

def p_expr_inlist(p):
    '''exprinlist : exprinlist COMMA expression
                  | exprinlist SEMICOLON expression
                  | exprinlist NEWLINE expression'''
    if p[2] in ['SEMICOLON', 'NEWLINE']:
        p[0] = '%s, OMPCSEMI, %s'%(p[1], p[3])
    else:
        p[0] = '%s, %s'%(p[1], p[3])

def p_expr_inlist2(p):
    '''exprinlist : exprinlist expression'''
    p[0] = '%s, %s'%(p[1], p[2])


def p_expr_inlist_token(p):
    '''exprinlist : exprinlist SEMICOLON
                  | exprinlist COMMA'''
    p[0] = p[1]

def p_statement_empty(p):
    '''statement : empty'''
    p[0] = ''

def p_expression_inlist_empty(p):
    "exprinlist : empty"
    p[0] = p[1]

def p_empty(p):
    "empty : "
    p[0] = ''

_pinlist = False
def p_expr_inlist_2(p):
    '''exprinlist : expression'''
    global _pinlist
    _pinlist = True
    p[0] = p[1]

def p_expression_binop(p):
    """expression : expression '+' expression
                  | expression '-' expression
                  | expression '*' expression
                  | expression '/' expression
                  | expression '\\\\' expression
                  | expression '^' expression
                  | expression DOTTIMES expression
                  | expression DOTDIVIDE expression
                  | expression DOTRDIVIDE expression
                  | expression DOTPOWER expression
                  | expression NOTEQUAL expression
                  | expression ISEQUAL expression
                  | expression LESS expression
                  | expression GREATER expression
                  | expression LESSEQUAL expression
                  | expression GREATEREQUAL expression
                  | expression ELAND expression
                  | expression ELOR expression
                  | expression AND expression
                  | expression OR expression"""
    if p[2] == '+'  : p[0] = '%s + %s'%(p[1], p[3])
    elif p[2] == '-'  : p[0] = '%s - %s'%(p[1], p[3])
    elif p[2] == '*'  : p[0] = '%s * %s'%(p[1], p[3])
    elif p[2] == '/'  : p[0] = '%s / %s'%(p[1], p[3])
    elif p[2] == '\\' : p[0] = '%s /ldiv/ %s'%(p[1], p[3])
    elif p[2] == '^'  : p[0] = '%s ** %s'%(p[1], p[3])
    elif p[2] == '.*' : p[0] = '%s *elmul* %s'%(p[1], p[3])
    elif p[2] == './' : p[0] = '%s /eldiv/ %s'%(p[1], p[3])
    elif p[2] == '.\\': p[0] = '%s /elldiv/ %s'%(p[1], p[3])
    elif p[2] == '.^' : p[0] = '%s **elpow** %s'%(p[1], p[3])
    # conditional and logical
    elif p[2] == '~=' : p[0] = '%s != %s'%(p[1], p[3])
    elif p[2] == '==' : p[0] = '%s == %s'%(p[1], p[3])
    elif p[2] == '<'  : p[0] = '%s < %s'%(p[1], p[3])
    elif p[2] == '>'  : p[0] = '%s > %s'%(p[1], p[3])
    elif p[2] == '<=' : p[0] = '%s <= %s'%(p[1], p[3])
    elif p[2] == '>=' : p[0] = '%s >= %s'%(p[1], p[3])
    elif p[2] == '&'  : p[0] = '%s & %s'%(p[1], p[3])
    elif p[2] == '|'  : p[0] = '%s | %s'%(p[1], p[3])
    elif p[2] == '&&' : p[0] = '%s and %s'%(p[1], p[3])
    elif p[2] == '||' : p[0] = '%s or %s'%(p[1], p[3])

def p_expression_not(p):
    "expression : NOT expression"
    p[0] = 'not %s'%p[2]

def p_expression_uminus(p):
    "expression : '-' expression %prec UMINUS"
    p[0] = '-%s'%p[2]

def p_expression_option(p):
    "cmd_option : '-' NAME"
    p[0] = '-%s'%p[2]

def p_expression_uplus(p):
    "expression : '+' expression %prec UPLUS"
    p[0] = p[2]

def p_expression_group(p):
    "expression : LPAREN exprlist RPAREN"
    p[0] = '(%s)'%p[2]

def p_expression_empty_group(p):
    "expression : NAME LPAREN RPAREN"
    p[0] = '%s()'%p[1]

def p_expr_mcat(p):
    'expression : exprmcat'
    #if p[1] == '[]'
    p[0] = 'mcat(%s)'%p[1]
    
def p_expression_list(p):
    """exprmcat : LBRACKET exprinlist RBRACKET"""
    global _pinlist
    _pinlist = False
    p[0] = '[%s]'%p[2]

def p_expression_cell(p):
    "expression : LCURLY exprinlist RCURLY"
    global _pinlist
    _pinlist = False
    p[0] = 'mcellarray([%s])'%p[2]

def p_expression_conjtranspose(p):
    'expression : expression CONJTRANS'
    p[0] = '%s.cT'%p[1]

def p_expression_transpose(p):
    'expression : expression TRANS'
    p[0] = '%s.T'%p[1]

def p_expression_string(p):
    "expression : STRING"
    p[0] = "mstring(%s)"%p[1]

def p_expression_indexflat(p):
    "indexflat : LPAREN ':' RPAREN"
    p[0] = '(mslice[:])'

def p_expr_flatslice(p):
    'expression : ":"'
    p[0] = 'mslice[:]'

def _check_name(name):
    from keyword import iskeyword
    if name == 'class':
        name = 'mclass'
    elif iskeyword(name) or isompcreserved(name):
        # FIXME ? maybe not
        # MATLAB does not allow names starting with '_', so we shuold be safe
        # prepending an underscore to the name of a variable
        name = '_' + name
    return name
    
def p_expression_sub_flat(p):
    "expression : NAME indexflat"
    p[0] = '%s%s'%(p[1], p[2])

def p_expression_sub(p):
    "name_sub : NAME LPAREN exprlist RPAREN"
    p[0] = '%s(%s)'%(p[1], p[3])

def p_name_attr2(p):
    """name_attr : name_sub '.' NAME
                 | name_attr '.' NAME
                 | name_attr '.' name_sub
                 | name_sub '.' name_sub"""
    p[0] = '%s.%s'%(p[1], p[3])

def p_name_attr(p):
    "name_attr : NAME"
    p[0] = '%s'%p[1]

def p_expression_attr(p):
    "expression : name_attr"
    p[0] = p[1]

def p_expression_sub2(p):
    """name_sub : NAME LCURLY exprinlist RCURLY"""
    p[0] = '%s(%s)'%(p[1], p[3])

def p_expression_items(p):
    "expression : name_sub"
    p[0] = '%s'%p[1]

def p_expression_slice(p):
    """slice : expression ':' expression ':' expression 
             | expression ':' expression"""
    if len(p) == 6:
        p[0] = '%s:%s:%s'%(p[1],p[3],p[5])
    else:
        p[0] = '%s:%s'%(p[1],p[3])

def p_expression_mslice(p):
    "expression : slice"
    p[0] = 'mslice[%s]'%p[1]

def p_expression_number(p):
    "expression : NUMBER"
    p[0] = p[1]

def p_expression_name(p):
    "expression : NAME"

_more = False
_lineno = 0
def p_error(p):
    global _comment, _more, _pinlist, _lineno, _last_line
    if p:
        if p.value == 'NEWLINE' and _pinlist:
            _more = True
        else:
            _print_error(_last_line)
            _print_error("Syntax error at line %d '%s'!" %(_lineno, p.value))
            pass
    else:
        if _pinlist:
            _more = True
        else:
            _print_error("Syntax error at EOF")

import yacc
yacc.yacc(debug=1)

def translate(data, outfile=sys.stdout):
    """Entry point to the OMPC translator.
    This function functions as a preprocessor. There are aspect of M-language
    that are difficult (cause conflicts) to be solved by a parser. It is also 
    much faster to implement some of the syntax by very simple checks.
    The preprocessor
     - combines continuations '...' (single line is submitted to the compiler)
     - removes comments, but makes it possible to add them later
     -
    """
    global _lineno, _last_line
    from re import sub, finditer
    com = ''
    d = []
    _lineno = 1
    for x in data.split('\n'):
        # preprocess, the returned values are strip of whitespace, and 
        # the optional coment is returned
        s, com = _ompc_preprocess(x)
        # if s is None a continuation was requested, submit the next line
        if s is None:
            continue
        
        _last_line = s
        yacc.myparse(s + '\n', outfile)
        
        # FIXME do something about the comments
        if s.strip():
            _print3000(_gettabs()[:-4] + com.strip(), file=outfile)
        else:
            _print3000(com, file=outfile)
        com = ''
        _lineno += 1

def translate_to_str(data):
    from StringIO import StringIO
    out = StringIO()
    translate(data, out)
    return out.getvalue()

_xbuf = ''
def _myparse(x, outfile=sys.stdout):
    global _more, _xbuf, _outfile, _last_line
    _outfile = outfile
    _last_line = _xbuf + x
    ret = yacc.parse(_xbuf + x)
    if _more:
        # this takes care of the newline inside of [] and {}. We don't want 
        # to have the newline as another token
        _xbuf += x.strip()
        if not _xbuf.endswith(';'):
            _xbuf += ';'
        _more = False
    else:
        _xbuf = ''
        more = False
    return ret

yacc.myparse = _myparse

# when searching for comments we make thigs easier by replacing contetns
# of all strings with something else than "%"
def _mysub(x):
    "Helper for replacement of strings."
    f, t = x.span()
    return 'x'*(t-f)

_cont = []
def _ompc_preprocess(x):
    """OMPC preprocessor.
    Takes a single line of m-code and returns a tuple of
    stripped m-code and a comment.
    Continuation is requested by the 1st returned value set to None.
    """
    global _cont, _pinlist
    from re import sub, findall, finditer
    # skip empty statements and take care of possible funny endlines 
    # only '\n' is allowed into the parser
    x = x.replace('\r', '')
    if not x.strip():
        return '', ''
    
    # remove comments
    x2 = sub(r"'((?:''|[^\n'])*)'", _mysub, x)
    pos = list(finditer(r'\s*%.*', x2))
    com = ''
    if pos:
        pos = pos[0].start()
        com = x[pos:].replace('%', '#', 1)
        x = x[:pos]
        if not x.strip():
            com = com.lstrip()
    
    # combine continuations
    _cont += [ x ]
    if x.strip().endswith('...'):
        _cont[-1] = x.strip()[:-3]
        return None, com
    
    # take care of lines like the following
    # "save a b c d -v7.3"
    # "hold"
    # these can be detected, they can not have '{}[]()=', they are simply
    # NAMEs and NUMBERs behind a name
    
    # FIXME should I make another parser just for this?
    LOC = ''.join(_cont)
    tname = t_NAME.__doc__
    tnum = t_NUMBER.__doc__
    if not _pinlist:
        from re import match
        mf2 = match(r'%(NAME)s\s+-?(%(NAME)s|%(NUMBER)s).*'% \
                            {'NAME':tname, 'NUMBER':tnum}, LOC)
        toks = LOC.split()
        #if len(findall(r'[(){}\[\]=]', LOC)) == 0 and \
        if mf2 is not None and \
                    toks and toks[0] not in _keywords:
            from re import split
            names = [ x for x in split('[;,\s]*', LOC.strip()) if x ]
            # names = LOC.split()
            LOC = '%s(%s)'%( names[0], ', '.join([ "'%s'"%x for x in names[1:] ]) )
    
    _cont = []
    return LOC, com


usage = """\
ompcply.py            - to get ompc compiler test console
ompcply.py lexdebug   - to get the console with debug output from tokenizer.
ompcply.py file.m     - will translate an m-file to OMPC .pym file.

The output is always to the standard output.
"""

__all__ = ['translate', 'yacc', 'lex']

if __name__ == "__main__":
    import sys, os
    LEXDEBUG = 0
    if len(sys.argv) > 1:
        if sys.argv[1] == 'lexdebug':
            LEXDEBUG = 1
        else:
            if not os.path.exists(sys.path[1]):
                print usage
            else:
                translate(open(sys.argv[1], 'U').read())
            sys.exit()
                
    print "Welcome to OMPC compiler test console!"
    print "(Ctrl+C to exit)"
    print
    
    # the ompc prompt loop, break with Ctrl+C
    _lineno = 1
    while 1:
        try:
            s = raw_input('ompc> ') + '\n'
        except EOFError:
            break
        
        # preprocess, the returned values are strip of whitespace, and 
        # the optional coment is returned
        s, com = _ompc_preprocess(s)
        # if s is None a continuation was requested, submit the next line
        if s is None:
            continue
        
        # if s is empty don't do anything
        if not s:
            continue
        
        if LEXDEBUG:
            # Tokenize
            lex.input(s)
            while 1:
                tok = lex.token()
                if not tok: break      # No more input
                print tok
                print _errors
                _errors = []
        
        yacc.myparse(s)
        print
