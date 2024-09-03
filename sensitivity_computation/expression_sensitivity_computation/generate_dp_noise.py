"""
@author: limou
@mail:  li-mou.li-mou@inria.fr
@date:  2023/3/20
@file: generate_dp_noise.py
"""

from abc import ABC, abstractmethod
from astailed import *
from compute_expression_sensitivity import compute_pure_vector_sensitivity, compute_matrix_expression_sensitivity, compute_dataset_expression_sensitivity


# compute noise
class DP_ProbabilityDistribution(ABC):
    def __int__(self, expression_domain, expr_str, dpvar_list, options, adjacency_notion, vector_dimension, vectors, substitution, scale, dataset_query) -> None:
        """
        Base class for the probability distribution under differential privacy

        :param expression_domain
        :param expr_str:
        :param dpvar_list:
        :param options:
        :param adjacency_notion:
        :param vector_dimension:
        :param vectors:
        :param substitution:
        :param scale:
        :return: None
        """
        self.expression_domain = expression_domain
        self.expr_str          = expr_str
        self.dpvar_list        = dpvar_list
        self.options           = options
        self.adjacency_notion  = adjacency_notion
        self.vector_dimension  = vector_dimension
        self.vectors           = vectors
        self.substitution      = substitution
        self.scale             = scale
        self.dataset_query     = dataset_query

    @property
    def sensitivity(self):
        if self.expression_domain == 'hadamard_vector_or_scalar':
            return compute_pure_vector_sensitivity(self.expr_str,
                                                   self.dpvar_list,
                                                   self.adjacency_notion,
                                                   self.vector_dimension,
                                                   self.vectors,
                                                   self.substitution,
                                                   self.scale)

        elif self.expression_domain == 'matrices':
            return compute_matrix_expression_sensitivity(self.expr_str,
                                                         self.adjacency_notion,
                                                         self.vector_dimension,
                                                         self.vectors,
                                                         self.substitution,
                                                         self.scale)

        elif self.expression_domain == 'dataset':
            return compute_dataset_expression_sensitivity(self.expr_str,
                                                          self.adjacency_notion,
                                                          self.dataset_query,
                                                          self.vectors,
                                                          self.scale)

    @abstractmethod
    def generate_noise(self):
        self.dp_metric_object = self.options['dp_metric']
        self.dp_mode = self.options['dp_mode']


class LaplaceDistribution(DP_ProbabilityDistribution):
    def __int__(self, expression_domain, expr_str, dpvar_list, options, adjacency_notion, vector_dimension, vectors, substitution, scale, dataset_query) -> None:
        """
        :return: None
        """
        super().__init__(expression_domain, expr_str, dpvar_list, options, adjacency_notion, vector_dimension, vectors, substitution, scale,  dataset_query)

    def generate_noise(self):
        """
        generate the Laplace noise
        :return:
        """
        epsilon = self.dp_metric_object.epsilon
        laplacian_noise_to_be_added = []
        sensitivity_dimension = self.sensitivity['matrice_dimension']

        if self.sensitivity['matrice_dimension'] == 'scalar' or self.sensitivity['matrice_dimension'] == 'vector':
            for i in range(sensitivity_dimension[0]):
                laplacian_noise_to_be_added.append(np.random.laplace(loc=0, scale=self.sensitivity[i] / epsilon))

        if self.sensitivity['matrice_dimension'] == 'matrix':
            for i in range(sensitivity_dimension[0]):
                row = []
                for j in range(sensitivity_dimension[1]):
                    row.append(np.random.laplace(loc=0, scale=self.sensitivity[i][j] / epsilon))
                laplacian_noise_to_be_added.append(row)

        return laplacian_noise_to_be_added


class GaussDistribution(DP_ProbabilityDistribution):
    def __int__(self, expression_domain, expr_str, dpvar_list, options, adjacency_notion, vector_dimension, vectors, substitution, scale, dataset_query) -> None:
        """
        :return: None
        """
        super().__init__(expression_domain, expr_str, dpvar_list, options, adjacency_notion, vector_dimension, vectors, substitution, scale, dataset_query)

    def generate_noise(self):
        """
        :return: generated gaussian noise according to Gaussian differential privacy
        """
        epsilon, delta = self.dp_metric_object.epsilon, self.dp_metric_object.delta
        gaussian_noise_to_be_added = []
        sensitivity_dimension = self.sensitivity['matrice_dimension']

        if self.sensitivity['matrice_dimension'] == 'scalar' or self.sensitivity['matrice_dimension'] == 'vector':
            for i in range(sensitivity_dimension[0]):
                sigma = np.sqrt(2 * np.log(1.25 / delta)) * self.sensitivity[i] / epsilon
                gaussian_noise_to_be_added.append(np.random.normal(loc=0, scale=sigma))

        if self.sensitivity['matrice_dimension'] == 'matrix':
            for i in range(sensitivity_dimension[0]):
                row = []
                for j in range(sensitivity_dimension[1]):
                    sigma = np.sqrt(2 * np.log(1.25 / delta)) * self.sensitivity[i][j] / epsilon
                    row.append(np.random.normal(loc=0, scale=sigma))
                gaussian_noise_to_be_added.append(row)

        return gaussian_noise_to_be_added


class RenyiDistribution(DP_ProbabilityDistribution):
    def __int__(self, expression_domain, expr_str, dpvar_list, options, adjacency_notion, vector_dimension, vectors, substitution, scale, dataset_query) -> None:
        """
        :return: None
        """
        super().__init__(expression_domain, expr_str, dpvar_list, options, adjacency_notion, vector_dimension, vectors, substitution, scale, dataset_query)

    def generate_noise(self):
        """
        :return: generated Gaussian noise according to Renyi differential privacy
        """
        alpha, epsilon_bar = self.dp_metric_object.alpha, self.dp_metric_object.epsilon_bar
        renyi_noise_to_be_added = []
        sensitivity_dimension = self.sensitivity['matrice_dimension']

        if self.sensitivity['matrice_dimension'] == 'scalar' or self.sensitivity['matrice_dimension'] == 'vector':
            for i in range(sensitivity_dimension[0]):
                sigma = np.sqrt((self.sensitivity[i] ** 2 * alpha) / (2 * epsilon_bar))
                renyi_noise_to_be_added.append(np.random.normal(loc=0, scale=sigma))

        if self.sensitivity['matrice_dimension'] == 'matrix':
            for i in range(sensitivity_dimension[0]):
                row = []
                for j in range(sensitivity_dimension[1]):
                    sigma = np.sqrt((self.sensitivity[i][j] ** 2 * alpha) / (2 * epsilon_bar))
                    row.append(np.random.normal(loc=0, scale=sigma))
                renyi_noise_to_be_added.append(row)

        return renyi_noise_to_be_added


ProbabilityDistributionClassedMap = {'laplace': LaplaceDistribution, 'gauss': GaussDistribution, 'renyi': RenyiDistribution}
