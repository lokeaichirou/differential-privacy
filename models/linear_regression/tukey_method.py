
from sklearn.linear_model import LinearRegression

from models.linear_regression.dp_linear_regression import DifferentiallyPrivateLinearRegression

from tailed_notions import *


def perturb_and_sort_matrix(input_matrix):
    """
    Perturbs and sorts input_matrix.

    :param input_matrix: Matrix in which each row is a vector to be perturbed and sorted.
    :return:             Adds a small amount of noise to each entry in input_matrix and then sorts each row in increasing order with shape of d*m.
    """
    d, m                = input_matrix.shape
    perturbation_matrix = 1e-10 * np.random.rand(d, m)
    perturbed_matrix    = input_matrix + perturbation_matrix
    return np.sort(perturbed_matrix, axis=1)


def log_measure_geq_all_depths(projections):
    """
    Computes log(volume) of region of at least depth, for all possible depths.

    :param projections: Matrix where each row is a projection of the data sorted in increasing order.
    :return:            Array A where A[i] is the (natural) logarithm of the volume in projections of the region of depth > i.
    """
    max_depth = int(np.ceil(len(projections[0]) / 2))  # d//2
    diff      = np.flip(projections, axis=1) - projections
    return np.sum(np.log(diff[:, :max_depth]), axis=0)


def racing_sample(log_terms):
    """
    Numerically stable method for sampling from an exponential distribution.

    :param log_terms: Array of terms of form log(coefficient) - (exponent term).
    :return:          A sample from the distribution over 0, 1, ..., len(log_terms) - 1 where integer k has probability
                      proportional to exp(log_terms[k]). For details, see the "racing sampling" described in https://arxiv.org/abs/2201.12333.
    """
    return np.argmin(np.log(np.log(1.0 / np.random.uniform(size=log_terms.shape))) - log_terms)


def restricted_racing_sample_depth(projections,
                                   epsilon,
                                   restricted_depth):
    """
    Executes epsilon-DP Tukey depth exponential mechanism on projections.

    :param projections:      Matrix where each row is a projection of the data sorted in increasing order.
    :param epsilon:          The output will be (epsilon, delta)-DP, where delta is determined by the preceding call to distance_to_unsafety.
    :param restricted_depth: Sampling will be restricted to points in projections of depth at least restricted_depth.
    :return:                 A sample from the epsilon-DP Tukey depth exponential mechanism on projections (line 3 in Algorithm 1 in https://arxiv.org/pdf/2106.13329.pdf).
    """

    projections            = projections[:, (restricted_depth-1):-(restricted_depth-1)]
    atleast_volumes        = np.exp(log_measure_geq_all_depths(projections))
    measure_exact_all      = atleast_volumes
    measure_exact_all[:-1] = atleast_volumes[:-1] - atleast_volumes[1:]
    depths                 = np.arange(restricted_depth, restricted_depth + len(atleast_volumes))
    log_terms              = np.log(measure_exact_all) + epsilon * depths

    # add 1 because returned depth is 0-indexed
    return 1 + racing_sample(log_terms)


def distance_to_unsafety(log_volumes,
                         epsilon,
                         delta,
                         t,
                         k_low,
                         k_high):
    """
    Returns Hamming distance lower bound computed by PTR check.

    :param log_volumes: Array of logarithmic volumes of regions of different depths; log_volumes[i] = log(volume of region of depth > i).
    :param epsilon:     The overall check is (epsilon, delta)-DP.
    :param delta:       The overall check is (epsilon, delta)-DP.
    :param t:           Fixed depth around which neighboring depth volumes are computed.
    :param k_low:       Lower bound for neighborhood size k. First call of this function should use k_low=-1
    :param k_high:      Upper bound for neighborhood size k.

    :return:            Hamming distance lower bound computed by PTR check.
    """
    k                 = int((k_low + k_high) / 2)
    log_vol_ytk       = log_volumes[t-k-2]
    log_vols_ytk_gs   = log_volumes[t+k+1:len(log_volumes)-1]
    log_epsilon_terms = (epsilon / 2) * np.arange(1, len(log_vols_ytk_gs) + 1)
    log_threshold     = np.log(delta / (8 * np.exp(epsilon)))

    if np.min((log_vol_ytk - log_vols_ytk_gs) - log_epsilon_terms) <= log_threshold:
        if k_low >= k_high - 1:
            return k_high
        else:
            new_k_low = k_low + int((k_high - k_low) / 2)
            return distance_to_unsafety(log_volumes, epsilon, delta, t, new_k_low, k_high)

    if k_high > k_low + 1:
        new_k_high  = k_high - int((k_high - k_low) / 2)
        return distance_to_unsafety(log_volumes, epsilon, delta, t, k_low, new_k_high)

    return k_low


def log_measure_geq_all_dims(depth,
                             projections):
    """
    Computes log(length) of region of at least depth, for each dimension.

    :param depth:       Desired depth in projections. Assumes 1 <= depth < len(projections[0]) / 2.
    :param projections: Matrix where each row is a projection of the data sorted in increasing order.

    :return:            Array A where A[j] is the (natural) logarithm of the length in projections of the region of
                        depth >= i in dimension j+1.
    """

    return np.log(projections[:, -depth] - projections[:, depth - 1])


def sample_geq_1d(depth,
                  projection):
    """
    Samples a point of at least given depth from projection.

    :param depth:      Lower bound on depth of point to sample. Assumes 1 <= depth <= len(proj) / 2.
    :param projection: Increasing array of 1-dimensional points.
    :return:           Point sampled uniformly at random from the region of at least the given depth in projection.
    """
    low  = projection[depth-1]
    high = projection[-depth]
    return np.random.uniform(low, high)


def sample_exact_1d(depth, projection):
    """
    Samples a point of exactly given depth from projection.

    :param depth:      Depth of point to sample. Assumes 1 <= depth <= len(proj) / 2.
    :param projection: Increasing array of 1-dimensional points.
    :return:           Point sampled uniformly at random from the region of given depth in projection.
    """

    left_low      = projection[depth-1]
    left_high     = projection[depth]
    right_low     = projection[-(depth+1)]
    right_high    = projection[-depth]
    measure_left  = left_high - left_low
    measure_right = right_high - right_low

    if np.random.uniform() < measure_left / (measure_left + measure_right):
      return left_low + np.random.uniform() * measure_left
    else:
      return right_low + np.random.uniform() * measure_right


def sample_exact(depth, projections):
    """
    Samples a point of exactly given depth from projections.

    :param depth:       Minimum depth of point to sample. Assumes 1 <= depth < len(proj) / 2.
    :param projections: Matrix of size (d, (m//2)) where each row is a projection of the data sorted in increasing order.
    :return:            Point sampled uniformly at random from the region of exactly given depth in projections.
    """

    d, _                            = projections.shape
    log_measures_greater_than_depth = log_measure_geq_all_dims(depth + 1, projections)
    log_measures_geq_depth          = log_measure_geq_all_dims(depth, projections)
    # exact_lengths[j] = W_{j+1, depth} in the paper's notation

    log_exact_lengths               = np.log(np.exp(log_measures_geq_depth) - np.exp(log_measures_greater_than_depth))

    # exp(log_volume_greater_than_depth_left[j]) = V_{<j+1, depth+1}, in the paper's notation.
    # log_volume_greater_than_depth_left[j] is the volume measured along the first j dimensions of the region of depth
    # greater than the depth argument to this function

    log_volume_greater_than_depth_left     = np.zeros(d)
    log_volume_greater_than_depth_left[1:] = np.cumsum(log_measures_greater_than_depth)[:-1]
    # exp(right_dims_geq_than_depth[j]) = V_{>j, depth}, in the paper's notation

    log_right_dims_geq_depth               = np.zeros(d)
    log_right_dims_geq_depth[:-1]          = (np.cumsum(log_measures_greater_than_depth[::-1])[::-1])[1:]
    log_volumes                            = log_exact_lengths + log_volume_greater_than_depth_left + log_right_dims_geq_depth
    sampled_volume_idx                     = racing_sample(log_volumes)

    # sampled point will be exactly depth in dimension sampled_volume_idx and >= depth elsewhere
    sample                                 = np.zeros(d)

    for j in range(sampled_volume_idx):
        sample[j] = sample_geq_1d(depth + 1, projections[j, :])

    sample[sampled_volume_idx] = sample_exact_1d(depth, projections[sampled_volume_idx, :])
    for j in range(sampled_volume_idx+1, d):
        sample[j] = sample_geq_1d(depth, projections[j, :])

    return sample


def multiple_regressions(features, labels, num_models):
    """
    Computes num_models regression models on random partition of given data.

    :param features:   Matrix of feature vectors. Assumed to have intercept feature.
    :param labels:     Vector of labels.
    :param num_models: Number of models to train.

    :return:           (num_models x d) matrix of models, where features is (n x d).
    :raises:           RuntimeError: [num_models] models requires [num_models * d] points, but given features only has [len(features)] points.
    """

    (n, d)     = features.shape
    batch_size = int(n / num_models)
    if batch_size < d:
        raise RuntimeError(str(num_models) + " models requires " + str(num_models * d) + " points, but given features only has " + str(n) + " points.")

    models            = np.zeros((num_models, d))
    for i in range(num_models):
        features = features.reset_index(drop=True)
        labels   = labels.reset_index(drop=True)

        features_batch = features.iloc[batch_size * i:batch_size * i + batch_size, :]
        labels_batch = labels.iloc[batch_size * i:batch_size * i + batch_size]

        model                   = LinearRegression()
        model.fit(features_batch, labels_batch)
        models[i, :]            = model.coef_

    return models


def dp_tukey(models,
             epsilon,
             delta):
    """
    Runs (epsilon, delta)-DP Tukey mechanism using models.

    :param models:  of size m*d, Feature vectors of non-private models. Assumes that each user contributes to a single model.
    :param epsilon: Computed model satisfies (epsilon, delta)-DP.
    :param delta:   Computed model satisfies (epsilon, delta)-DP.
    :return:        Regression model computed using Tukey mechanism, or 0 if PTR fails.
    """

    projections = perturb_and_sort_matrix(models.T)  # d*m
    max_depth   = int(len(models) / 2)  # m//2

    # compute log(volume_i)_{i=1}^max_depth where volume_i is the volume of the region of depth >= i, according to projections
    log_volumes = log_measure_geq_all_depths(projections)
    t           = int(max_depth / 2)  # m//4

    # do ptr check
    split_epsilon = epsilon / 2
    distance      = distance_to_unsafety(log_volumes, split_epsilon, delta, t, -1, t-1)
    threshold     = np.log(1 / (2 * delta)) / split_epsilon
    if not distance + np.random.laplace(scale=split_epsilon) > threshold:
        print("ptr fail")
        return np.zeros(len(models[0]))

    # sample a depth using the restricted exponential mechanism
    depth         = restricted_racing_sample_depth(projections, split_epsilon, t)

    # sample uniformly from the region of given depth
    return sample_exact(depth, projections[:, t:-t])


class TukeyBasedDPLinearRegression(DifferentiallyPrivateLinearRegression):

    def __init__(self,
                 privacy_notion,
                 num_models,
                 accountant = None,
                 **unused_args):

        """
        Linear Regression with differential privacy algorithm class based on adassp algorithm [1].

        :param privacy_notion
        :param bounds_X:      tuple type.          Bounds of the data, provided as a tuple of the form (min, max). min
                                                   and max can either be scalars, covering the min/max of the entire
                                                   data, or vectors with one entry per feature.  If not provided, the
                                                   bounds are computed on the data when .fit() is first called,
                                                   resulting in class PrivacyLeakWarning
        :param bounds_y:      tuple type.          Same as bounds_X, but for the training label set y.
        :param fit_intercept: bool, default: True. Whether to calculate the intercept for this model. If set False, no
                                                   intercept will be used in computations (i.e. data is expected to be
                                                   centered).
        :param copy_X:        bool, default: True. If True, X will be copied; else, it may be overwritten.
        :param random_state:  int or RandomState, optional. Controls the randomness of the model. To obtain a
                                                            deterministic behaviour during randomisation, random_state
                                                            has to be fixed to an integer.
        :param accountant:    BudgetAccountant, optional. Accountant to keep track of privacy budget.

        Reference:
        [1] Amin, K., Joseph, M., Ribero, M., & Vassilvitskii, S. (2022). Easy Differentially Private Linear Regression.
        ArXiv, abs/2208.07353.
        """

        super().__init__(privacy_notion = privacy_notion,
                         accountant     = accountant)

        self.num_models = num_models

    def fit(self, X, y, sample_weight=None):
        """
        Fit linear model.

        :param X:                   array-like or sparse matrix, shape (n_samples, n_features). Training data
        :param y:                   array_like, shape (n_samples, n_targets).                   Target values.
        :param scale_training_data: whether strictly follow the assumption on training data distribution in [1]
        :param sample_weight:
        :return:                    self : returns an instance of self.
        """
        random_state = self.necessary_steps(X, y, sample_weight)

        models       = multiple_regressions(features   = X,
                                            labels     = y,
                                            num_models = self.num_models)

        self.coef_   = dp_tukey(models, self.epsilon, self.delta)

        if y.ndim == 1:
            self.coef_ = np.ravel(self.coef_)

        self._set_intercept(None, None, None)

        # Deduct the privacy cost from the privacy budget
        # -------------------------------------------------------------------------------------------------------------
        self.accountant.spend(self.epsilon, self.delta)

        return self
