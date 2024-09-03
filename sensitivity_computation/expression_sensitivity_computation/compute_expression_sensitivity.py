"""
@author: limou
@mail:  li-mou.li-mou@inria.fr
@date:  2023/3/13
@file: compute_expression_sensitivity.py

For vector box adjacency notion, generate the variable_domain_map for each variable and each dimension, and then call
ast_to_possible_range in compute_expression_range.py. to compute the sensitivity value for each dimension.
All the variables in the expression should have same dimension.

For vector diff adjacency notion,

"""
import copy

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))

import sys
sys.path.insert(0, current_dir)
sys.path.append(parent_dir)
print(sys.path)

from ...tailed_notions import *

from compute_expression_range import *
from compute_expression_derivative import *

import matrice_expression_operation
import dataset_expression_operation

# from matrice_expression_operation import *
# from dataset_expression_operation import *


def ast_to_sensitivity(e, expr, adjacency_notions_list, substitution, scale):
    """

    :param e: ast.Expression node
    :param expr
    :param adjacency_notions_list: each dp variable has an adjacency notion instance
    :param substitution: {'y': 1}
    :return:
    """
    dimension = 0
    variable_domain_map = dict()

    # iterate over differential privacy variable(s) and fill variable_domain_map
    for adjacency_notion in adjacency_notions_list:
        # For the vector box adjacency, generate the variable_domain_map as the input to the ast_to_possible_range
        # function
        variable_name = adjacency_notion.dp_variable
        min_vector    = adjacency_notion.get_min_vector()
        max_vector    = adjacency_notion.get_max_vector()
        dimension     = adjacency_notion.vector_dimension

        if dimension == 1:
            if not variable_domain_map.get(0):
                variable_domain_map[0] = {}
            variable_domain_map[0][variable_name] = [min_vector, max_vector]
        else:
            for i in range(dimension):
                if not variable_domain_map.get(i):
                    variable_domain_map[i] = {}
                variable_domain_map[i][variable_name] = [min_vector[i], max_vector[i]]

    # compute sensitivity value(s) for vector(scalar) in terms of the VectorBoxAdjacency
    if isinstance(adjacency_notions_list[0], VectorBoxAdjacency):
        # iterate over dimension(s) and get sensitivity value for each dimension
        sensitivity_values_for_all_dimensions_in_vector_box_adjacency_notion = []
        for i in range(dimension):
            expression_range = ast_to_possible_range(e                   =e,
                                                     original_expr_str   =expr,
                                                     variable_domain_map =variable_domain_map[i],
                                                     substitution        =substitution)[0]
            sensitivity_values_for_all_dimensions_in_vector_box_adjacency_notion.append(abs(expression_range[1] - expression_range[0]))

        return sensitivity_values_for_all_dimensions_in_vector_box_adjacency_notion

    # compute sensitivity value(s) for vector(scalar) in terms of the VectorDiffAdjacency
    elif isinstance(adjacency_notions_list[0], VectorDiffAdjacency):
        sensitivity_values_for_all_dimensions_in_vector_diff_adjacency_notion = []
        for i in range(dimension):
            sensitivity_values_for_all_dimensions_in_vector_diff_adjacency_notion.append(
                expr_to_ast_vector_diff(expr_str            =expr,
                                        variable_domain_map =variable_domain_map[i],
                                        scale               =scale,
                                        substitution        =substitution)
            )

        return sensitivity_values_for_all_dimensions_in_vector_diff_adjacency_notion


def compute_pure_vector_sensitivity(expr_str, dp_var_list, adjacency_notion, vector_dimension, vectors, substitution, scale, print_info=False):
    """

    :param expr_str: '2*x*x'
    :param dp_var_list: ['x']
    :param adjacency_notion: 'VectorBoxAdjacency', 'VectorDiffAdjacency'
    :param vector_dimension: every variable has same dimension
    :param vectors: {'x':[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]}, each differential variable has a list, within which the
                    first tuple contains the minimum values for each dimension and the second tuple contains the maximum
                    values for each dimension
    :param substitution: {'y': 1}
    :param scale
    :return:
    """

    # assign an adjacency notion instance for each variable
    adjacency_notions_list = []

    for dp_var in dp_var_list:

        # broadcasting may be used
        if not isinstance(vectors[dp_var][0], tuple):
            if vector_dimension > 1:
                min_val, max_val = vectors[dp_var][0], vectors[dp_var][1]
                vectors[dp_var] = [tuple((min_val for _ in range(vector_dimension))),
                                   tuple((max_val for _ in range(vector_dimension)))]
        # broadcasting won't be used
        else:
            if len(vectors[dp_var][0]) != vector_dimension:
                raise GIVEN_VECTOR_CAN_NOT_HAVE_GIVEN_VECTOR_DIMENSION

        if adjacency_notion == 'VectorBoxAdjacency':
            adj = VectorBoxAdjacency(dp_var, vector_dimension)
            adj.set_vector_box(vectors[dp_var][0], vectors[dp_var][1])
            adjacency_notions_list.append(adj)

        elif adjacency_notion == 'VectorDiffAdjacency':
            adj = VectorDiffAdjacency(dp_var, vector_dimension)
            adj.set_vector_box_and_scale(vectors[dp_var][0], vectors[dp_var][1], scale)
            adjacency_notions_list.append(adj)

    collected_ast_nodes = ast.parse(expr_str, '<string>', 'eval')
    vector_sensitivity  = ast_to_sensitivity(collected_ast_nodes, expr_str, adjacency_notions_list, substitution, scale)

    if len(vector_sensitivity) > 1:
        matrice_type      = 'vector'
        matrice_dimension = (len(vector_sensitivity), 1)
    else:
        matrice_type      = 'scalar'
        matrice_dimension = (1, 1)

    sensitivity = {'sensitivity_value': copy.deepcopy(vector_sensitivity),
                   'matrice_type':      matrice_type,
                   'matrice_dimension': matrice_dimension}

    if print_info:
        print('expr_to_sensitivity(', expr_str, ') = ',
               sensitivity,
              'with dp variable (', dp_var_list, ')'
              'and adjacency_notion of (', adjacency_notion ,')',
              'in the range (', vectors,')')

    return sensitivity


def compute_matrix_expression_sensitivity(expr_str, adjacency_notion, vector_dimension, vectors, substitution, scale, print_info=False):
    """

    :param expr_str:
    :param adjacency_notion:
    :param vector_dimension:
    :param vectors:
    :return:
    """
    returned_map, domain_map = matrice_expression_operation.expr_to_ast_matrix_operation(expr_str, vector_dimension, vectors)

    if returned_map['type'] == 'scalar':
        row = returned_map['rows'][0]
        sensitivity_value = compute_pure_vector_sensitivity(expr_str         =row,
                                                            dp_var_list      =returned_map['variables'],
                                                            adjacency_notion =adjacency_notion,
                                                            vector_dimension =1,
                                                            vectors          =domain_map,
                                                            substitution     =substitution,
                                                            scale            =scale)['sensitivity_value']

        sensitivity_value_map = {'sensitivity_value': sensitivity_value,
                                 'matrice_type':      'scalar',
                                 'matrice_dimension': (1, 1)}

        print('expr_to_sensitivity(', expr_str, ') = ',
              sensitivity_value_map,
              'with adjacency_notion of (', adjacency_notion, ')',
              'in the range (', vectors, ')')

        return sensitivity_value_map

    elif returned_map['type'] == 'vector':
        vector_expression_sensitivity = []
        for row in returned_map['rows']:
            vector_expression_sensitivity.append(compute_pure_vector_sensitivity(expr_str        =row,
                                                                                dp_var_list      =returned_map['variables'],
                                                                                adjacency_notion =adjacency_notion,
                                                                                vector_dimension =1,
                                                                                vectors          =domain_map,
                                                                                substitution     =substitution,
                                                                                scale            =scale)
                                             )
        sensitivity_value_map = {'sensitivity_value': vector_expression_sensitivity,
                                 'matrice_type':      'vector',
                                 'matrice_dimension': returned_map['size']}

        print('expr_to_sensitivity(', expr_str, ') = ',
              sensitivity_value_map,
              'with adjacency_notion of (', adjacency_notion, ')',
              'in the range (', vectors, ')')

        return sensitivity_value_map

    elif returned_map['type'] == 'tr_vector':
        vector_expression_sensitivity = []
        for col in returned_map['columns']:
            vector_expression_sensitivity.append(compute_pure_vector_sensitivity(expr_str         =col,
                                                                                 dp_var_list      =returned_map['variables'],
                                                                                 adjacency_notion =adjacency_notion,
                                                                                 vector_dimension =1,
                                                                                 vectors          =domain_map,
                                                                                 substitution     =substitution,
                                                                                 scale            =scale)
                                                 )
        sensitivity_value_map = {'sensitivity_value': vector_expression_sensitivity,
                                 'matrice_type':      'vector',
                                 'matrice_dimension': returned_map['size'][::-1]}

        print('expr_to_sensitivity(', expr_str, ') = ',
              sensitivity_value_map,
              'with adjacency_notion of (', adjacency_notion, ')',
              'in the range (', vectors, ')')

        return sensitivity_value_map

    elif returned_map['type'] == 'matrix':
        matrix_expression_sensitivity = []
        for row in returned_map['rows']:
            rows_sensitivity = []
            for element in row:
                rows_sensitivity.append(compute_pure_vector_sensitivity(expr_str         =element,
                                                                        dp_var_list      =returned_map['variables'],
                                                                        adjacency_notion =adjacency_notion,
                                                                        vector_dimension =1,
                                                                        vectors          =domain_map,
                                                                        substitution     =substitution,
                                                                        scale            =scale)
                                        )
            matrix_expression_sensitivity.append(rows_sensitivity)

        sensitivity_value_map = {'sensitivity_value': matrix_expression_sensitivity,
                                 'matrice_type':      'matrix',
                                 'matrice_dimension': returned_map['size']}

        print('expr_to_sensitivity(', expr_str, ') = ',
              sensitivity_value_map,
              'with adjacency_notion of (', adjacency_notion, ')',
              'in the range (', vectors, ')')

        return sensitivity_value_map


def dataset_sensitivity_based_on_dataset_query(adjacency_notion, dataset_query, num_instances):
    """

    :param adjacency_notion:
    :param dataset_query: 'sum(A[:,2])'
    :return:
    """

    query_function                       = dataset_query.split('(')[0]
    queried_dataset_name_and_instance_id = dataset_query.split('(')[1]
    queried_dataset_name                 = queried_dataset_name_and_instance_id.split('[')[0]
    queried_instance_id                  = int(queried_dataset_name_and_instance_id.split('[')[1][2])

    if isinstance(adjacency_notion, DatasetReplaceVectorBoxAdjacency):
        min_vector, max_vector =  adjacency_notion.get_min_vector(), adjacency_notion.get_max_vector()
        if query_function in ['sum', 'max', 'min', 'median']:
            return max_vector[queried_instance_id] - min_vector[queried_instance_id]
        elif query_function == 'mean':
            return (max_vector[queried_instance_id] - min_vector[queried_instance_id])/num_instances
        elif query_function == 'count':
            return 1

    elif isinstance(adjacency_notion, DatasetReplaceVectorNormAdjacency):
        scale = adjacency_notion.get_scale()
        if query_function in ['sum', 'max', 'min', 'median']:
            return scale
        elif query_function == 'mean':
            return scale/num_instances
        elif query_function == 'count':
            return 1


def compute_dataset_expression_sensitivity(expr_str, adjacency_notion, dataset_query, vectors, scale, print_info=False):
    """

    :param expr_str:
    :param adjacency_notion:
    :param dataset_query: sum(A[:,2])
    :param vectors: {'d1':[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)], 'd2'}, each dataset has a list, within which the
                    first tuple contains the minimum values for each dimension and the second tuple contains the maximum
                    values for each dimension
    :param scale:  for vector diff adjacency
    :return:
    """

    collected_ast_nodes = ast.parse(expr_str, '<string>', 'eval')
    dataset_map         = dataset_expression_operation.ast_to_dataset(collected_ast_nodes)
    dataset_name        = dataset_map['dataset_name']
    num_instances       = dataset_map['dataset_dimensions'][0]
    instance_dimension  = dataset_map['dataset_dimensions'][1]
    if len(vectors[dataset_name][0]) != instance_dimension:
        raise DATASET_INSTANCES_NUMBER_NOT_EQUAL_TO_BOX_VECTOR_DIMENSION

    if adjacency_notion == 'DatasetReplaceVectorBoxAdjacency':
        adj = DatasetReplaceVectorBoxAdjacency(dataset_map['dataset_name'], instance_dimension)
        adj.set_vector_box(vectors[dataset_name][0], vectors[dataset_name][1])

    elif adjacency_notion == 'DatasetReplaceVectorNormAdjacency':
        adj = DatasetReplaceVectorNormAdjacency(dataset_map['dataset_name'], instance_dimension)
        adj.set_order_and_scale(2, scale)

    dataset_sensitivity   = dataset_sensitivity_based_on_dataset_query(adj, dataset_query, num_instances)
    sensitivity_value_map = {'sensitivity_value': dataset_sensitivity,
                             'matrice_type':      'scalar',
                             'matrice_dimension': (1,1)}

    print('expr_to_sensitivity(', expr_str, ') = ',
          sensitivity_value_map,
          'with adjacency_notion of (', adjacency_notion, ')',
          'in the range (', vectors, ')')

    return sensitivity_value_map


# single variable of 1 dimension and to the power of 2; VectorBoxAdjacency
compute_pure_vector_sensitivity(expr_str         ="2*x*x",
                                dp_var_list      =['x'],
                                adjacency_notion ='VectorBoxAdjacency',
                                vector_dimension =1,
                                vectors          ={'x': [0, 1]},
                                substitution     ={},
                                scale            =None,
                                print_info       =True)

# single variable of 1 dimension and to the power of 2; VectorDiffAdjacency
compute_pure_vector_sensitivity(expr_str         ="2*x*x",
                                dp_var_list      =['x'],
                                adjacency_notion ='VectorDiffAdjacency',
                                vector_dimension =1,
                                vectors          ={'x': [0, 2]},
                                substitution     ={},
                                scale            =1,
                                print_info       =True)

# single variable of 3 dimensions and to the power of 2; VectorBoxAdjacency
compute_pure_vector_sensitivity(expr_str          ='2*x*x',
                                dp_var_list       =['x'],
                                adjacency_notion  ='VectorBoxAdjacency',
                                vector_dimension  =3,
                                vectors           ={'x': [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]},
                                substitution      ={},
                                scale             =None,
                                print_info        =True)

# single variable of 3 dimensions(broadcasting) and to the power of 2; VectorBoxAdjacency
compute_pure_vector_sensitivity(expr_str          ='2*x*x',
                                dp_var_list       =['x'],
                                adjacency_notion  ='VectorBoxAdjacency',
                                vector_dimension  =3,
                                vectors           ={'x': [0.0, 1.0]},
                                substitution      ={},
                                scale             =None,
                                print_info        =True)

# multi-variable of 1 dimension and to the power of 2 with substitution; VectorBoxAdjacency
compute_pure_vector_sensitivity(expr_str         ="2*x*x+3*y*y",
                                dp_var_list      =['x'],
                                adjacency_notion ='VectorBoxAdjacency',
                                vector_dimension =1,
                                vectors          ={'x': [0, 1]},
                                substitution     ={"y": 3.0},
                                scale            =None,
                                print_info       =True)

# multi-variable of 1 dimension and to the power of 2 with substitution; VectorDiffAdjacency
compute_pure_vector_sensitivity(expr_str         ="2*x*x+3*y*y",
                                dp_var_list      =['x'],
                                adjacency_notion ='VectorDiffAdjacency',
                                vector_dimension =1,
                                vectors          ={'x': [0, 2]},
                                substitution     ={"y": 3.0},
                                scale            =1,
                                print_info       =True)

# multi-variable of 1 dimension and to the power of 2; VectorBoxAdjacency
compute_pure_vector_sensitivity(expr_str         ="2*x*x+3*y*y",
                                dp_var_list      =['x', 'y'],
                                adjacency_notion ='VectorBoxAdjacency',
                                vector_dimension =1,
                                vectors          ={'x': [0, 1], "y": [0, 1]},
                                substitution     ={},
                                scale            =None,
                                print_info       =True)

# multi-variable of 1 dimension and to the power of 2; VectorDiffAdjacency
compute_pure_vector_sensitivity(expr_str         ="2*x*x+3*y*y",
                                dp_var_list      =['x', 'y'],
                                adjacency_notion ='VectorDiffAdjacency',
                                vector_dimension =1,
                                vectors          ={'x': [0, 1], "y": [0, 1]},
                                substitution     ={},
                                scale            =1,
                                print_info       =True)

# single variable of 3 dimensions in the form of matrix * vector; VectorBoxAdjacency
compute_matrix_expression_sensitivity(expr_str         ='matrix((1,2,3),(4,5,6))*x',
                                      adjacency_notion ='VectorBoxAdjacency',
                                      vector_dimension ={'x': 3},
                                      vectors          ={'x': [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]},
                                      substitution     ={},
                                      scale            =None,
                                      print_info       =True)

# single variable of 3 dimensions in the form of matrix * vector; VectorDiffAdjacency
compute_matrix_expression_sensitivity(expr_str         ='matrix((1,2,3),(4,5,6))*x',
                                      adjacency_notion ='VectorDiffAdjacency',
                                      vector_dimension ={'x': 3},
                                      vectors          ={'x': [(0.0, 0.0, 0.0), (3.0, 3.0, 3.0)]},
                                      substitution     ={},
                                      scale            =2,
                                      print_info       =True)

# single variable of 3 dimensions in the form of transposed_vector * matrix * vector; VectorBoxAdjacency
compute_matrix_expression_sensitivity(expr_str         ='tr(x)*matrix((1,2,3),(4,5,6),(7,8,9))*x',
                                      adjacency_notion ='VectorBoxAdjacency',
                                      vector_dimension ={'x': 3},
                                      vectors          ={'x': [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]},
                                      substitution     ={},
                                      scale            =None,
                                      print_info       =True)

# single variable of 3 dimensions in the form of transposed_vector * matrix * vector; VectorDiffAdjacency
compute_matrix_expression_sensitivity(expr_str         ='tr(x)*matrix((1,2,3),(4,5,6),(7,8,9))*x',
                                      adjacency_notion ='VectorDiffAdjacency',
                                      vector_dimension ={'x': 3},
                                      vectors          ={'x': [(0.0, 0.0, 0.0), (3.0, 3.0, 3.0)]},
                                      substitution     ={},
                                      scale            =2,
                                      print_info       =True)

# single variable of 3 dimensions in the form of vector * transposed_vector; VectorBoxAdjacency
compute_matrix_expression_sensitivity(expr_str         ='x*tr(x)',
                                      adjacency_notion ='VectorBoxAdjacency',
                                      vector_dimension ={'x': 3},
                                      vectors          ={'x': [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]},
                                      substitution     ={},
                                      scale            =None,
                                      print_info       =True)

# single variable of 3 dimensions in the form of vector * transposed_vector; VectorDiffAdjacency
compute_matrix_expression_sensitivity(expr_str         ='x*tr(x)',
                                      adjacency_notion ='VectorDiffAdjacency',
                                      vector_dimension ={'x': 3},
                                      vectors          ={'x': [(0.0, 0.0, 0.0), (2.0, 2.0, 2.0)]},
                                      substitution     ={},
                                      scale            =2,
                                      print_info       =True)


compute_dataset_expression_sensitivity(expr_str         ='dataset(d(5,3))',
                                       adjacency_notion ='DatasetReplaceVectorBoxAdjacency',
                                       dataset_query    ='sum(A[:,2])',
                                       vectors          ={'d':[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]},
                                       scale            =None,
                                       print_info       =True)
