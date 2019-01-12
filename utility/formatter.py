import copy


def convert_function(f):
    """
    Return the expression in a human readable string.
    """

    string = ""
    stack = []
    for node in f:
        stack.append((node, []))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            string = reformat_primitives(prim, args)
            if len(stack) == 0:
                break  # If stack is empty, all nodes should have been seen
            stack[-1][1].append(string)
    return string


def reformat_primitives(prim, args):

    """
    Convert inverse prims according to:
    [Dd]iv(a,b) -> Mul[a, 1/b]
    [Ss]ub(a,b) -> Add[a, -b]
    We achieve this by overwriting the corresponding format method of the sub and div prim.
    """

    prim = copy.copy(prim)

    converter = {
        'sub': lambda *args_: "Add({}, Mul(-1,{}))".format(*args_),
        'div': lambda *args_: "Mul({}, Pow({}, -1))".format(*args_),
        'mul': lambda *args_: "Mul({},{})".format(*args_),
        'add': lambda *args_: "Add({},{})".format(*args_)
    }
    prim_formatter = converter.get(prim.name, prim.format)

    return prim_formatter(*args)
