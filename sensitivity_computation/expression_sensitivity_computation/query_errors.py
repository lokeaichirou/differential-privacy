"""
@author: limou
@mail:  li-mou.li-mou@inria.fr
@date:  2023/2/23
@file: query_errors.py
"""

VARIABLE_WITHOUT_DEFINED_DOMAIN = 231
INAPPROPRIATE_BASE_NUMBER = 232
NON_MONOTONICAL = 233
MORE_THAN_ONE_VARIABLE_IN_ANY_MULTIPLIER = 234
NOT_SUPPORT_COMPLICATED_MULTIPLICATION = 235
NOT_SUPPORT_BASE_NUMBER_AS_VARIABLE_OR_FUNCTION = 236
EXPRESSION_EXCEPT_FOR_SINGLE_VARIABLE_NOT_ALLOWED_AS_INPUT_TO_SINUSOIDAL_FUNCTION = 237
INPUT_VARIABLE_TO_SINUSOIDAL_FUNCTION_SHOULD_FOLLOW_NAIVE_BOX_ADJACENCY = 238
UNSUPPORTED_SUBSTITUTION = 239
DEMOMINATOR_IS_ZERO = 240
ZERO_IN_DEMOMINATOR_RANGE = 241
GIVEN_VECTOR_CAN_NOT_HAVE_GIVEN_VECTOR_DIMENSION = 242
SHAPES_DO_NOT_MATCH_IN_MATRIX_OR_VECTOR_MULTIPLICATION = 243
DATASET_INSTANCES_NUMBER_NOT_EQUAL_TO_BOX_VECTOR_DIMENSION = 244
FUNCTIONS_INVOLVED_DO_NOT_HAVE_DERIVATIVES_THAT_ARE_MONOTONICALLY_INCREASING_OR_DECREASING = 245
SCALE_SET_FOR_VECTOR_DIFF_ADJACENCY_SMALLER_THAN_DOMAIN_INTERVAL = 246

class QueryException(Exception):
    def __init__(self, code: int, *arr_params):
        self.code = code
        self.arr_params = arr_params

    def __str__(self):
        """

        Returns:
            str: string when print(e)
        """
        if self.code == VARIABLE_WITHOUT_DEFINED_DOMAIN:
            return f"Variable {self.arr_params}'s domain is not defined in the variable_domain_map"

        elif self.code == INAPPROPRIATE_BASE_NUMBER:
            return f"Inappropriate base number{self.arr_params}"

        elif self.code == NON_MONOTONICAL:
            return f"Non-monotonical equations"

        elif self.code == MORE_THAN_ONE_VARIABLE_IN_ANY_MULTIPLIER:
            return f"There are more than one variable in the multiplier"

        elif self.code == NOT_SUPPORT_COMPLICATED_MULTIPLICATION:
            return f"It does NOT support variable * function, function * variable, function * function."

        elif self.code == NOT_SUPPORT_BASE_NUMBER_AS_VARIABLE_OR_FUNCTION:
            return f"It does not support base number as variable or function"

        elif self.code == EXPRESSION_EXCEPT_FOR_SINGLE_VARIABLE_NOT_ALLOWED_AS_INPUT_TO_SINUSOIDAL_FUNCTION:
            return f"Input to sinusoidal function only supports single variable"

        elif self.code == INPUT_VARIABLE_TO_SINUSOIDAL_FUNCTION_SHOULD_FOLLOW_NAIVE_BOX_ADJACENCY:
            return f"Input variable to sinusoidal function should have [0,1] domain"

        elif self.code == UNSUPPORTED_SUBSTITUTION:
            return f"Unsupported substitution"

        elif self.code == DEMOMINATOR_IS_ZERO:
            return f"Denominator is equal to zero"

        elif self.code == ZERO_IN_DEMOMINATOR_RANGE:
            return f"Zero in denominator range"

        elif self.code == GIVEN_VECTOR_CAN_NOT_HAVE_GIVEN_VECTOR_DIMENSION:
            return f"Given vector has different dimension from the given vector dimension"

        elif self.code == SHAPES_DO_NOT_MATCH_IN_MATRIX_OR_VECTOR_MULTIPLICATION:
            return f"Shapes do not match in matrix or vector multiplication"

        elif self.code == DATASET_INSTANCES_NUMBER_NOT_EQUAL_TO_BOX_VECTOR_DIMENSION:
            return f"Dataset instances number not equal to the box vector dimension"

        elif self.code == FUNCTIONS_INVOLVED_DO_NOT_HAVE_DERIVATIVES_THAT_ARE_MONOTONICALLY_INCREASING_OR_DECREASING:
            return f"The functions involved in the expression should have derivatives that are both monotonically " \
                   f"increasing or decreasing,or some of the derivatives are monotonically increasing or decreasing " \
                   f"while others are constant"

        elif self.code == SCALE_SET_FOR_VECTOR_DIFF_ADJACENCY_SMALLER_THAN_DOMAIN_INTERVAL:
            return f"The scale set for vector diff adjacency is smaller than the domain interval"
