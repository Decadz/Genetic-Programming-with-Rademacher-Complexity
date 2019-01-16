def division(left, right):

    """
    Special protected division that is safe for X/0.

    :param left: First Param
    :param right: Second Param
    :return: Value
    """

    try:  # Need to cast to float! to avoid Nan/Inf.
        return float(left)/float(right)
    except ZeroDivisionError:
        return 1
