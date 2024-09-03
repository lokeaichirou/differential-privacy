"""
@author: limou
@mail:  li-mou.li-mou@inria.fr
@date:  2023/3/20
@file: compute_expression_derivative.py

Please write in the form of x*x+2*x+y instead of x*(x+2)*y, can not have addition operation with constant in bracket!
The expression must present in expanded form!

It supports up to the power of 2, e.g. x^2, i.e. x*x is allowed, while x^3 is not allowed

The functions involved in the expression should have derivatives that are both monotonically increasing or decreasing,
or some of the derivatives are monotonically increasing or decreasing while others are constant, e.g. x+log(x) is not
allowed because x has derivative of 1 and log(x) has derivative of 1/x, their derivatives are both monotonically
increasing. However, x*x+log(x) is not allowed because x*x has derivative of 2*x which is monotonically increasing while
log(x) has derivative of 1/x which is monotonically decreasing.

square root is not being supported

"""

import ast
from ast_utils import *
from compute_expression_range import *
from query_errors import *


def check_whether_functions_in_derivative_value(value, list_of_functions):
    """

    :param value: '-1*log'
    :param list_of_functions: ['log', 'log2', 'log10']
    :return:
    """

    for function in list_of_functions:
        if function in value:
            return True
    return False


def scale_check(variable_domain_map, scale):

    for variable, domain in variable_domain_map.items():
        if abs(domain[1] - domain[0]) < scale:
            return False
    return True


def legal_check(derivative_map):

    check_map = {'increasing': None, 'decreasing': None}
    for derivative_variable in derivative_map.keys():
        if derivative_variable == 'constant':
            continue
        if derivative_variable not in ['log', 'log2', 'log10']:
            check_map['decreasing'] = True
        else:
            check_map['increasing'] = True

    if check_map['decreasing'] == True and check_map['increasing'] == True:
        return False

    return True


def merge_dict_by_plus(x, y):
    for k, v in x.items():
        if k in y.keys():
            y[k] = str(y[k]) + '+' + str(v)
        else:
            y[k] = v

    return y


def merge_dict_by_subtract(x, y):
    for k, v in x.items():
        if k in y.keys():
            y[k] = str(y[k]) + '-' + str(v)
        else:
            y[k] = v

    return y


def merge_dict(x,y):
    for k, v in x.items():
        if k in y.keys():
            y[k] += v
        else:
            y[k] = v

    return y


def ast_to_derivative_constant(e):
    return {'constant': e.value}


def ast_to_derivative_unaryop(e, original_expr_str, variable_domain_map):
    """
    Dealing with unary operator

    :param e: a unary operator node, i.e. ast.UnaryOP object
    :return: negative constant, e.g. -2
    """

    if isinstance(e.op, ast.USub):
        if not isinstance(e.operand, ast.Constant):
            rephrased_expr_str  = '-1*' + original_expr_str[1:]
            collected_ast_nodes = ast.parse(rephrased_expr_str, '<string>', 'eval')
            return ast_to_possible_derivative(collected_ast_nodes, original_expr_str, variable_domain_map)
        else:
            return {'constant': -e.operand.value}, str(-e.operand.value)


def function_operation(e, cur_expr_str, variable_domain_map):
    """
    Deal the function as a newly created variable and modify the original form of the function in the expression to be a
    newly created variable, e.g. 'log(x-1)' -> log_1, its domain is the range of the function
    :param e:
    :param cur_expr_str:
    :param variable_domain_map:
    :return: derivative_map, newly_created_string
    """

    global number_of_functions_in_the_expression
    number_of_functions_in_the_expression += 1

    expr_range, variable_contained = ast_to_possible_range(e.args[0], cur_expr_str, variable_domain_map, {})

    function_type = e.func.id
    new_variable = function_type + '_' + str(number_of_functions_in_the_expression)
    variable_domain_map[new_variable] = expr_range

    # {'log_1': 'log'}, 'log_1'
    return {new_variable: function_type}, new_variable


def ast_to_derivative_binop(e, original_expr_str, variable_domain_map):
    """

    :param e:
    :param original_expr_str:
    :param variable_domain_map:
    :return:
    """

    left_node_map,  left_new_string  = ast_to_possible_derivative(e.left,  original_expr_str, variable_domain_map)
    right_node_map, right_new_string = ast_to_possible_derivative(e.right, original_expr_str, variable_domain_map)
    # variable_contained = list(set(variable_contained_in_left_node) | set(variable_contained_in_right_node))

    # For addition
    if isinstance(e.op, ast.Add):
        return merge_dict_by_plus(left_node_map, right_node_map),     left_new_string + '+' + right_new_string

    # For subtraction
    elif isinstance(e.op, ast.Sub):
        return merge_dict_by_subtract(left_node_map, right_node_map), left_new_string + '-' + right_new_string

    # For multiplication
    if isinstance(e.op, ast.Mult):
        # 2*x
        if 'constant' in left_node_map:
            for key, value in right_node_map.items():

                # {'x': 1}
                if isinstance(value, int) or isinstance(value, float):
                    right_node_map[key] = left_node_map['constant'] * value
                # {'x': '2*x'}
                else:
                    right_node_map[key] = str(left_node_map['constant']) + '*' + str(value)

            return right_node_map, left_new_string + '*' + right_new_string

        # x*2
        elif 'constant' in right_node_map:
            for key, value in left_node_map.items():
                if isinstance(value, int) or isinstance(value, float):
                    left_node_map[key] = right_node_map['constant'] * value
                else:
                    left_node_map[key] = str(right_node_map['constant']) + '*' + str(value)

            return left_node_map, left_new_string + '*' + right_new_string

        else:
            # x*x
            if list(left_node_map.keys())[0] == list(right_node_map.keys())[0]:
                variable_name = list(left_node_map.keys())[0]
                return {variable_name: '2*'+ str(left_node_map[variable_name]) + '*' + str(right_node_map[variable_name]) + '*' + variable_name}, left_new_string + '*' + right_new_string

            # x*y
            return merge_dict(left_node_map, right_node_map), left_new_string + '*' + right_new_string


def ast_to_possible_derivative(e, cur_expr_str, variable_domain_map):
    """

    :param e:
    :param cur_expr_str:
    :param variable_domain_map:
    :return:
    """

    # For identifying as an expression node
    if isinstance(e, ast.Expression):
        return ast_to_possible_derivative(e.body, cur_expr_str, variable_domain_map)

    # For identifying e as a variable name node
    elif isinstance(e, ast.Name):
        # {'x': 1}, 'x'
        return {e.id: 1}, e.id

    # For identifying e as a constant node
    elif ast_is_constant(e):
        return ast_to_derivative_constant(e), str(e.value)

    # For identifying as an unary node
    elif isinstance(e, ast.UnaryOp):
        return ast_to_derivative_unaryop(e, cur_expr_str, variable_domain_map)

    # For identifying as a binary operator node
    elif isinstance(e, ast.BinOp):
        return ast_to_derivative_binop(e, cur_expr_str, variable_domain_map)

    # For identifying as a function node like log(x), log2(x), ...
    elif isinstance(e, ast.Call):
        return function_operation(e, cur_expr_str, variable_domain_map)

    ast_to_dp_error(e)
    return 0


def expr_to_ast_vector_diff(expr_str, variable_domain_map, scale, substitution, print_info=False):
    """

    :param expr_str:
    :param variable_domain_map:
    :param scale:
    :param substitution:
    :param print_info
    :return:
    """

    if not scale_check(variable_domain_map, scale):
        return SCALE_SET_FOR_VECTOR_DIFF_ADJACENCY_SMALLER_THAN_DOMAIN_INTERVAL

    global number_of_functions_in_the_expression
    number_of_functions_in_the_expression      = 0
    substitution_for_min, substitution_for_max = {}, {}

    collected_ast_nodes        = ast.parse(expr_str, '<string>', 'eval')
    derivative_map, new_string = ast_to_possible_derivative(collected_ast_nodes, expr_str, variable_domain_map)

    if not legal_check(derivative_map):
        raise FUNCTIONS_INVOLVED_DO_NOT_HAVE_DERIVATIVES_THAT_ARE_MONOTONICALLY_INCREASING_OR_DECREASING

    for key, value in derivative_map.items():
        if key == 'constant':
            continue

        # when derivative is monotonically decreasing value over the domain
        elif isinstance(value, str) and check_whether_functions_in_derivative_value(value, ['log', 'log2', 'log10']):
            substitution_for_min[key] = variable_domain_map[key][0]
            substitution_for_max[key] = variable_domain_map[key][0] + scale

        elif key in substitution:
            substitution_for_min[key] = substitution[key]
            substitution_for_max[key] = substitution[key]

        # when derivative is constant or monotonically increasing value over the domain, e.g. x, x*x, exp()
        else:
            substitution_for_min[key] = variable_domain_map[key][1] - scale
            substitution_for_max[key] = variable_domain_map[key][1]

    min_value = expr_to_ast_range(new_string, variable_domain_map, substitution_for_min)
    if isinstance(min_value, list):
        min_value = min_value[0]

    max_value = expr_to_ast_range(new_string, variable_domain_map, substitution_for_max)
    if isinstance(max_value, list):
        max_value = max_value[0]

    max_vector_diff = abs(max_value - min_value)

    if print_info:
        print('The max vector diff for', expr_str,
              '=', max_vector_diff,
              'with domain:', variable_domain_map,
              'and scale:', scale)

    return max_vector_diff
