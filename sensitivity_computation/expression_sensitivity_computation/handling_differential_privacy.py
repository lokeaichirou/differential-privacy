"""
@author: limou
@mail:  li-mou.li-mou@inria.fr
@date:  2023/3/28
@file: handling_differential_privacy.py


"""

from generate_dp_noise import *


class DifferentialPrivacyHandler:
    def __int__(self, noise_mechanism, expr_str, dpvar_list, options, adjacency_notion, vector_dimension, vectors, substitution, scale, **kwargs) -> None:
        """

        :param noise_mechanism:
        :param expr_str:
        :param dpvar_list:
        :param options:
        :param adjacency_notion:
        :param vector_dimension:
        :param vectors:
        :param substitution:
        :param scale:
        :return:
        """
        self.noise_mechanism   = noise_mechanism
        self.expr_str          = expr_str
        self.dpvar_list        = dpvar_list
        self.options           = options
        self.adjacency_notion  = adjacency_notion
        self.vector_dimension  = vector_dimension
        self.vectors           = vectors
        self.substitution      = substitution
        self.scale             = scale
        self.order             = kwargs.get('order', None)
        self.dataset_query     = kwargs.get('dataset_query', None)
        self.expression_domain = determine_type_of_expression_domain(self.expr_str)

    def generate_dp_noise(self):
        dp_probability_distribution = ProbabilityDistributionClassedMap[self.noise_mechanism](self.expression_domain,
                                                                                              self.expr_str,
                                                                                              self.dpvar_list,
                                                                                              self.options,
                                                                                              self.adjacency_notion,
                                                                                              self.vector_dimension,
                                                                                              self.vectors,
                                                                                              self.substitution,
                                                                                              self.scale,
                                                                                              self.dataset_query)
        noise_to_be_added = dp_probability_distribution.generate_noise()
        return noise_to_be_added


def determine_type_of_expression_domain(expr_str):
    """

    :param expr_str:
    :return:
    """
    if 'matrix' in expr_str or 'tr' in expr_str:
        expression_domain = 'matrices'
    elif 'dataset' in expr_str:
        expression_domain = 'dataset'
    else:
        expression_domain = 'hadamard_vector_or_scalar'

    return expression_domain
