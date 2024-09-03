"""
@author: limou
@mail:  li-mou.li-mou@inria.fr
@date:  2023/3/09
@file: compute_expression_range.py

Generate the numerical range of expression for vector box adjacency notion, given the variable(s) domain.
Variable in the expression is 1-dimensional.

We do not consider the potentially complicated case like non-monotonically increasing or decreasing functions yet,
what we are currently dealing with by the AST method are monotonically increasing or decreasing function given the
variable domain, meaning all the functions in the expression/sub-expression are monotonically increasing or
decreasing simultaneously! E.g. f(x) = 2*x, the domain should be either in negative range or positive range.

Apart from a*x, where a is a constant, x*x, constant^{variable or function}, other functions being supported include
square root, logarithm and exponential.

"""

import sys
import ast
from ast_utils import *
import numpy as np
import math
from query_errors import *

functions_map = {'sqrt': math.sqrt, 'log': math.log, 'log2': math.log2, 'log10': math.log10, 'exp': math.exp}


def ast_to_dp_error(e):
    print('ast_to_dp problem: ', e.__class__.__name__)
    ast.dump(e)


def ast_derive_constant(e):
    """

    :param e: a constant node
    :return: ast.Constant.value, []
    """
    if float(sys.version[:3]) <= 3.7:
        return e.value, []
    return e.value, []


def absolute_of_variable(result):
    """

    :param result:
    :return: [min_of_absolute_val_of_variable, max_of_absolute_val_of_variable]
    """
    if result[1] < 0:
        return [abs(result[1]), abs(result[0])]

    elif result[0] < 0 and result[1] >= 0:
        abs_a, abs_b = abs(result[0]), abs(result[1])
        max_val = abs_a if abs_a > abs_b else abs_b
        return [0, max_val]
    else:
        return [abs(result[0]), abs(result[1])]


def ast_to_variable_range(e, variable_domain_map, substitution):
    """
    Get the range of variable defined in variable_domain_map

    :param e: variable name node, i.e. ast.Name object
    :param variable_domain_map: {x: [x_min_value, x_max_value], y: [y_min_value, y_max_value], ...}
    :param substitution: {y: 3.0}
    :return: [min_value, max_value], [ variable_name ]
    """
    if e.id in substitution:
        return substitution[e.id], []
    elif e.id in variable_domain_map:
        return variable_domain_map[e.id], [e.id]
    else:
        raise QueryException(VARIABLE_WITHOUT_DEFINED_DOMAIN, e.id)


def ast_to_range_unaryop(e, original_expr_str, variable_domain_map, substitution):
    """
    Dealing with unary operator

    :param e: a unary operator node, i.e. ast.UnaryOP object
    :return: negative constant, e.g. -2
    """
    if isinstance(e.op, ast.USub):

        if not isinstance(e.operand, ast.Constant):
            rephrased_expr_str  = '-1*' + original_expr_str[1:]
            collected_ast_nodes = ast.parse(rephrased_expr_str, '<string>', 'eval')
            return ast_to_possible_range(collected_ast_nodes, original_expr_str, variable_domain_map, substitution)
        else:
            return -e.operand.value, []


def function_operation(e, original_expr_str, variable_domain_map, substitution):
    """
    Dealing with functions defined in functions_map and abs()

    :param e: a function node, ast.Call
    :param variable_domain_map: {x: [x_min_value, x_max_value], y: [y_min_value, y_max_value], ...}
    :return: [min_value, max_value], [ variable_contained_in_the_(sub)expression ]
    """

    function_type = e.func.id
    args_result, variable_contained = ast_to_possible_range(e.args[0], original_expr_str, variable_domain_map, substitution)

    if function_type == 'abs':
        return absolute_of_variable(args_result), variable_contained
    elif function_type == 'sin' or function_type == 'cos':
        if isinstance(e.args[0], ast.Name):
            if variable_domain_map[e.args[0].id][0] != 0 or variable_domain_map[e.args[0].id][1] != 1:
                raise INPUT_VARIABLE_TO_SINUSOIDAL_FUNCTION_SHOULD_FOLLOW_NAIVE_BOX_ADJACENCY
            else:
                if function_type == 'sin':
                    return [0, math.sin(1)]
                elif function_type == 'cos':
                    return [1, math.cos(1)]
        else:
            raise EXPRESSION_EXCEPT_FOR_SINGLE_VARIABLE_NOT_ALLOWED_AS_INPUT_TO_SINUSOIDAL_FUNCTION

    else:
        return [functions_map[function_type](args_result[0]), functions_map[function_type](args_result[1])], variable_contained


def ast_to_range_binop(e, original_expr_str, variable_domain_map, substitution):
    """
    Dealing with binary operator

    :param e: a binary operator node, i.e. ast.BinOp object
    :param original_expr_str
    :param variable_domain_map: {x: [x_min_value, x_max_value], y: [y_min_value, y_max_value], ...}
    :return: [min_value, max_value], [ variable_contained_in_the_(sub)expression ]
    """
    # 0 means list, 1 means constant
    list_or_constant_indicator = {'left': 0, 'right': 0}

    left_node_result,  variable_contained_in_left_node  = ast_to_possible_range(e.left,  original_expr_str, variable_domain_map, substitution)
    right_node_result, variable_contained_in_right_node = ast_to_possible_range(e.right, original_expr_str, variable_domain_map, substitution)
    variable_contained = list(set(variable_contained_in_left_node) | set(variable_contained_in_right_node))

    if not isinstance(left_node_result, list):
        list_or_constant_indicator['left'] = 1
    if not isinstance(right_node_result, list):
        list_or_constant_indicator['right'] = 1

    # For addition
    if isinstance(e.op, ast.Add):
        #   We stipulate the multiplication operations only present in such forms:
        #    (a) constant           +  variable/function
        #    (c) variable/function  +  constant
        #    (d) variable/function  +  variable/function

        # constant + [min_value, max_value]
        if list_or_constant_indicator['left'] == 1 and list_or_constant_indicator['right'] == 0:
            return [left_node_result + right_node_result[0], left_node_result + right_node_result[1]], variable_contained

        # [min_value, max_value] + constant
        elif list_or_constant_indicator['left'] == 0 and list_or_constant_indicator['right'] == 1:
            return [left_node_result[0] + right_node_result, left_node_result[1] + right_node_result], variable_contained

        # [min_value, max_value] + [min_value, max_value]
        elif list_or_constant_indicator['left'] == 0 and list_or_constant_indicator['right'] == 0:
            return [left_node_result[0] + right_node_result[0], left_node_result[1] + right_node_result[1]], variable_contained

        # constant + constant is allowed for the case of VectorDiff adjacency
        else:
            return [left_node_result + right_node_result, left_node_result + right_node_result], variable_contained

    # For subtraction
    if isinstance(e.op, ast.Sub):
        # constant - [min_value, max_value]
        if list_or_constant_indicator['left'] == 1 and list_or_constant_indicator['right'] == 0:
            return [left_node_result - right_node_result[0], left_node_result - right_node_result[1]], variable_contained

        # [min_value, max_value] - constant
        elif list_or_constant_indicator['left'] == 0 and list_or_constant_indicator['right'] == 1:
            return [left_node_result[0] - right_node_result, left_node_result[1] - right_node_result], variable_contained

        # [min_value, max_value] - [min_value, max_value] is not allowed, because we stipulate the functions in the
        # expression/sub-expression are simultaneously monotonically increasing or decreasing given the variable domain
        elif list_or_constant_indicator['left'] == 0 and list_or_constant_indicator['right'] == 0:
            raise QueryException(NON_MONOTONICAL)

        # constant - constant is allowed for the case of VectorDiff adjacency
        else:
            return [left_node_result - right_node_result, left_node_result - right_node_result], variable_contained

    # For multiplication
    if isinstance(e.op, ast.Mult):
        #   We stipulate the multiplication operations only present in such forms:
        #    (a) constant              *  constant
        #    (b) constant              *  variable or function
        #    (c) variable or function  *  constant
        #    (d) variable_1            *  variable_2
        #   It does NOT support variable * function, function * variable, function * function.

        if len(variable_contained_in_left_node) > 1 or len(variable_contained_in_right_node) > 1:
            raise QueryException(MORE_THAN_ONE_VARIABLE_IN_ANY_MULTIPLIER)

        # constant * constant
        if list_or_constant_indicator['left'] == 1 and list_or_constant_indicator['right'] == 1:
            return left_node_result * right_node_result, variable_contained

        # constant * [min_value, max_value]
        elif list_or_constant_indicator['left'] == 1 and list_or_constant_indicator['right'] == 0:
            constant = left_node_result
            if constant >= 0:
                return [constant * right_node_result[0], constant * right_node_result[1]], variable_contained
            else:
                return [constant * right_node_result[1], constant * right_node_result[0]], variable_contained

        # [min_value, max_value] * constant
        elif list_or_constant_indicator['left'] == 0 and list_or_constant_indicator['right'] == 1:
            constant = right_node_result
            if constant >= 0:
                return [constant * left_node_result[0], constant * left_node_result[1]], variable_contained
            else:
                return [constant * left_node_result[1], constant * left_node_result[0]], variable_contained

        # [min_value, max_value] * [min_value, max_value]
        elif list_or_constant_indicator['left'] == 0 and list_or_constant_indicator['right'] == 0:
            if (not isinstance(e.left, ast.Call)) and (not isinstance(e.right, ast.Call)):

                # Same variable in variable_1 * variable_2
                if variable_contained_in_left_node[0] == variable_contained_in_right_node[0]:
                    if left_node_result[0] > 0:
                        return [left_node_result[0] * right_node_result[0], left_node_result[1] * right_node_result[1]], variable_contained_in_left_node
                    else:
                        a = left_node_result[0] * right_node_result[0]
                        b = left_node_result[1] * right_node_result[1]
                        max_value = a if a>b else b
                        return [0, max_value], variable_contained_in_left_node

                # Different variables in variable_1 * variable_2
                else:
                    if left_node_result[0] > 0 and right_node_result[0] > 0:
                        return [left_node_result[0] * right_node_result[0], left_node_result[1] * right_node_result[1]], variable_contained
                    else:
                        ac = left_node_result[0] * right_node_result[0]
                        ad = left_node_result[0] * right_node_result[1]
                        bc = left_node_result[1] * right_node_result[0]
                        bd = left_node_result[1] * right_node_result[1]

                        mutual_multiplication_results = [ac, ad, bc, bd]
                        sorted_mutual_multiplication_results = sorted(mutual_multiplication_results)
                        return sorted_mutual_multiplication_results[2:], variable_contained
            else:
                raise QueryException(NOT_SUPPORT_COMPLICATED_MULTIPLICATION)

    # For division
    if isinstance(e.op, ast.Div):
        # (a) (function or variable)/constant
        if list_or_constant_indicator['left'] == 0 and list_or_constant_indicator['right'] == 1:
            constant = right_node_result
            if constant == 0:
                raise DEMOMINATOR_IS_ZERO
            elif constant > 0:
                return [left_node_result[0]/constant, left_node_result[1]/constant], variable_contained
            else:
                return [left_node_result[1]/constant, left_node_result[0]/constant], variable_contained

        # (b) constant / (function or variable)
        elif list_or_constant_indicator['left'] == 1 and list_or_constant_indicator['right'] == 0:
            constant = left_node_result
            if (right_node_result[0] <= 0 and right_node_result[1] > 0) or (right_node_result[0] < 0 and right_node_result[1] >= 0):
                raise ZERO_IN_DEMOMINATOR_RANGE
            if constant >= 0:
                return [constant/right_node_result[1], constant/right_node_result[0]], variable_contained
            else:
                return [constant/right_node_result[0], constant/right_node_result[1]], variable_contained

    # For power operation
    # (1) constant^{variable or function}
    if isinstance(e.op, ast.Pow):
        if list_or_constant_indicator['left'] == 1 and list_or_constant_indicator['right'] == 0:
            if left_node_result <= 0:
                raise QueryException(INAPPROPRIATE_BASE_NUMBER, left_node_result)
            elif left_node_result < 1:
                return [left_node_result ** right_node_result[1], left_node_result ** right_node_result[0]], variable_contained
            else:
                return [left_node_result ** right_node_result[0], left_node_result ** right_node_result[1]], variable_contained
        else:
            raise NOT_SUPPORT_BASE_NUMBER_AS_VARIABLE_OR_FUNCTION

    ast_to_dp_error(e)
    return 0


def ast_to_possible_range(e, original_expr_str, variable_domain_map, substitution):
    """
    Get the possible range of every sub-expresion

    :param e: iterated ast object node that is obtained by parsing the original expression string

    :param variable_domain_map: {x: [x_min_value, x_max_value], y: [y_min_value, y_max_value], ...}
    :param substitution: {y: 3.0}
    :return: [min_value_of_expression, max_value_of_expression], [ variable_contained_in_the_(sub)expression ]
    """

    # For identifying as an expression node
    if isinstance(e, ast.Expression):
        return ast_to_possible_range(e.body, original_expr_str, variable_domain_map, substitution)

    # For identifying e as a variable name node
    elif isinstance(e, ast.Name):
        return ast_to_variable_range(e, variable_domain_map, substitution)

    # For identifying e as a constant node
    elif ast_is_constant(e):
        return ast_derive_constant(e)

    # For identifying as an unary node
    elif isinstance(e, ast.UnaryOp):
        return ast_to_range_unaryop(e, original_expr_str, variable_domain_map, substitution)

    # For identifying as a binary operator node
    elif isinstance(e, ast.BinOp):
        return ast_to_range_binop(e, original_expr_str, variable_domain_map, substitution)

    # For identifying as a function node like log(x), log2(x), ...
    elif isinstance(e, ast.Call):
        return function_operation(e, original_expr_str, variable_domain_map, substitution)

    ast_to_dp_error(e)
    return 0


def expr_to_ast_range(expr_str, variable_domain_map, substitution):
    """
    Compute the possible domain of expression given the domain of variable.

    We do not consider the potentially complicated case like non-monotonically increasing or decreasing functions yet,
    what we are currently dealing with by the AST method are monotonically increasing or decreasing function given the
    variable domain, meaning all the functions in the expression/sub-expression are monotonically increasing or
    decreasing simultaneously!

    :param expr_str: original expression string in str format
    :param variable_domain_map: {x: [x_min_value, x_max_value], y: [y_min_value, y_max_value], ...}
    :param substitution: {y: 3.0}
    :return: [min_value_of_expression, max_value_of_expression]
    """
    collected_ast_nodes            = ast.parse(expr_str, '<string>', 'eval')
    expr_range, variable_contained = ast_to_possible_range(collected_ast_nodes, expr_str, variable_domain_map, substitution)
    return expr_range


def expr_to_ast_range_for_print_purpose(expr_str, variable_domain_map, substitution):
    """

    :param expr_str: original expression string in str format
    :param variable_domain_map: {x: [x_min_value, x_max_value], y: [y_min_value, y_max_value], ...}
    :param substitution: {y: 3.0}
    :return: [min_value_of_expression, max_value_of_expression]
    """
    collected_ast_nodes            = ast.parse(expr_str, '<string>', 'eval')
    expr_range, variable_contained = ast_to_possible_range(collected_ast_nodes, expr_str, variable_domain_map, substitution)
    print('expr_to_posible_range(', expr_str, ') = ', expr_range,
          'with variable_domain_map: ', variable_domain_map,
          'and substitution map: ', substitution)


# expr_to_ast_range_for_print_purpose(expr_str="log(x+5)", variable_domain_map={'x': [1, 2]}, substitution={})
# expr_to_ast_range_for_print_purpose(expr_str="log(x)", variable_domain_map={'x': [1, 2]}, substitution={})

#expr_to_ast_range_for_print_purpose(expr_str="dataset(d1:(m, n), d2:(p,q))", variable_domain_map={}, substitution={})
# expr_to_ast_range_for_print_purpose(expr_str="dataset(d, (5, 3))", variable_domain_map={}, substitution={})

# expr_to_ast_range_for_print_purpose(expr_str="1*x_0+2*x_1+3*x_2", variable_domain_map={'x_0': [0, 1], 'x_1': [0, 1], 'x_2': [0,1]}, substitution={})

# expr_to_ast_range_for_print_purpose(expr_str="x/2", variable_domain_map={'x': [0, 1]}, substitution={})
# expr_to_ast_range_for_print_purpose(expr_str="(x*x+y*y)/2", variable_domain_map={'x': [0, 1], 'y': [0, 1]}, substitution={})
# expr_to_ast_range_for_print_purpose(expr_str="2/x", variable_domain_map={'x': [1, 2]}, substitution={})

# expr_to_ast_range_for_print_purpose(expr_str="-2*x", variable_domain_map={'x': [0, 1], 'y': [0, 1]}, substitution={})
# expr_to_ast_range_for_print_purpose(expr_str="5*x+7", variable_domain_map={'x': [0, 1]}, substitution={})
# expr_to_ast_range_for_print_purpose(expr_str="x*x+7", variable_domain_map={'x': [0, 2]}, substitution={})

# expr_to_ast_range_for_print_purpose(expr_str="2*x+3*y", variable_domain_map={'x': [0, 1], 'y': [0, 1]}, substitution={})
# expr_to_ast_range_for_print_purpose(expr_str="2*x+3*y", variable_domain_map={'x': [0, 1]}, substitution={'y': 1})

# expr_to_ast_range_for_print_purpose(expr_str="2*x*y", variable_domain_map={'x': [0, 1], 'y': [0, 1]}, substitution={})

# expr_to_ast_range_for_print_purpose(expr_str="2**x", variable_domain_map={'x': [0, 2]}, substitution={})
# expr_to_ast_range_for_print_purpose(expr_str="0.5**x", variable_domain_map={'x': [0, 2]}, substitution={})

# expr_to_ast_range_for_print_purpose(expr_str="sqrt(x)", variable_domain_map={'x': [0, 3], 'y': [-4, 4]}, substitution={})
# expr_to_ast_range_for_print_purpose(expr_str="sqrt(x*x+y*y)", variable_domain_map={'x': [0, 3], 'y': [-4, 4]}, substitution={})
# expr_to_ast_range_for_print_purpose(expr_str="exp(x*x+y*y)", variable_domain_map={'x': [0, 3], 'y': [-4, 4]}, substitution={})
# expr_to_ast_range_for_print_purpose(expr_str="log2(abs(x)+2)", variable_domain_map={'x': [-14, 14]}, substitution={})

# expr_to_ast_range_for_print_purpose(expr_str="log2(x)+2*x", variable_domain_map={'x': [1, 3]}, substitution={})
