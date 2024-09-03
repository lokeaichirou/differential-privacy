"""
@author: limou
@mail:  li-mou.li-mou@inria.fr
@date:  2023/3/20
@file: ast_utils.py
"""

import sys
import ast
import numpy as np
import math
from query_errors import *


def ast_to_dp_error(e):
    print('ast_to_dp problem: ', e.__class__.__name__)
    ast.dump(e)


def ast_is_constant(e):
    """

    :param e: a node
    :return: True/False
    """
    if float(sys.version[:3]) <= 3.7:
        return isinstance(e, ast.Num)
    return isinstance(e, ast.Constant)


def ast_to_range_unaryop(e):
    """
    Dealing with unary operator

    :param e: a unary operator node, i.e. ast.UnaryOP object
    :return: negative constant, e.g. -2
    """
    if isinstance(e.op, ast.USub):
        return -e.operand.value, []
