def findTerminal(pset, expr, type_, stack, depth):
    term = random.choice(pset.terminals[type_])
    if isclass(term):
        term = term()
    expr.append(term)

def findPrimitive(pset, expr, type_, stack, depth):
    prim = random.choice(pset.primitives[type_])
    expr.append(prim)
    for arg in reversed(prim.args):
        stack.append((depth+1, arg))

def generate(pset, min_, max_, condition, type_=None):
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        (firstSearch, secondSearch) = (findTerminal, findPrimitive) if condition(height, depth) else (findPrimitive, findTerminal)

        try:
            firstSearch(pset, expr, type_, stack, depth)
        except:
            try:
                secondSearch(pset, expr, type_, stack, depth)
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The gp.generate function tried to add "\
                                 "a primitive or terminal of type '%s', but "\
                                 "there is none available." % (type_,)).with_traceback
    return expr
