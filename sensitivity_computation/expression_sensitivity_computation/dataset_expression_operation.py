"""
@author: limou
@mail:  li-mou.li-mou@inria.fr
@date:  2023/3/29
@file: dataset_expression_operation.py

dataset(d(m, n))
"""

import ast
from ast_utils import *


def dataset_operation(e):
    """

    :param e:
    :return: dataset_map = {'dataset_name':      'd'
                            'total_dimensions':   2
                            'dataset_dimensions': (5,3)
                            }
    """
    function_type = e.func.id
    #if function_type == 'dataset':
    #    dataset_map = {'dataset_name': e.args[0].func.id}
    #    dataset_total_dimensions = len(e.args[1].elts)
    #    dataset_map['total_dimensions'] = dataset_total_dimensions

    #    dataset_dimensions = []
    #    for i in range(dataset_total_dimensions):
    #        dataset_dimensions.append(e.args[1].elts[i].value)
    #    dataset_map['dataset_dimensions'] = tuple(dataset_dimensions)

    #    return dataset_map

    if function_type == 'dataset':
        dataset_map                     = {'dataset_name': e.args[0].func.id}
        dataset_total_dimensions        = len(e.args[0].args)
        dataset_map['total_dimensions'] = dataset_total_dimensions

        dataset_dimensions = []
        for i in range(dataset_total_dimensions):
            dataset_dimensions.append(e.args[0].args[i].n)
        dataset_map['dataset_dimensions'] = tuple(dataset_dimensions)

        return dataset_map


def ast_to_dataset(e):

    # For identifying as an expression node
    if isinstance(e, ast.Expression):
        return ast_to_dataset(e.body)

    # For identifying as a function node
    elif isinstance(e, ast.Call):
        return dataset_operation(e)

    ast_to_dp_error(e)
    return 0
