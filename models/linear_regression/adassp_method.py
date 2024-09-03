
from models.linear_regression.dp_linear_regression import DifferentiallyPrivateLinearRegression

from tailed_notions import *


def adassp(features, labels, epsilon, delta, rho=0.05):
    """
    Returns model computed using AdaSSP DP linear regression which is Algorithm 2 in <Revisiting differentially private
    linear regression: optimal and adaptive prediction & estimation in unbounded domain> https://arxiv.org/pdf/1803.02596.pdf

    :param features: Matrix of feature vectors. Assumed to have intercept feature.
    :param labels:   Vector of labels.
    :param epsilon:  Computed model satisfies (epsilon, delta)-DP.
    :param delta:    Computed model satisfies (epsilon, delta)-DP.
    :param rho:      Failure probability. The default of 0.05 is the one used in https://arxiv.org/pdf/1803.02596.pdf.
    :return:         Vector of regression coefficients. AdaSSP is described in Algorithm 2 of https://arxiv.org/pdf/1803.02596.pdf.
    """

    _, d           = features.shape

    # these bounds are data-dependent and not dp
    bound_x        = np.amax(np.linalg.norm(features, axis=1))
    bound_y        = np.amax(np.abs(labels))

    # Line 2 in Algorithm 2
    lambda_min     = max(0, np.amin(np.linalg.eigvals(np.matmul(features.T, features))))
    z              = np.random.normal(size=1)
    sensitivity    = np.sqrt(np.log(6 / delta)) / (epsilon / 3)
    private_lambda = max(0, lambda_min + sensitivity * (bound_x**2) * z - (bound_x**2) * np.log(6 / delta) / (epsilon / 3))

    # Line 3 in Algorithm 2
    final_lambda   = max(0, np.sqrt(d * np.log(6 / delta) * np.log(2 * (d**2) / rho)) * (bound_x**2) / (epsilon / 3) - private_lambda)

    # Line 4 in Algorithm 2
    # Generate symmetric noise_matrix where each upper entry is iid N(0,1)
    noise_matrix = np.random.normal(size=(d, d))
    noise_matrix = np.triu(noise_matrix)
    noise_matrix = noise_matrix + noise_matrix.T - np.diag(np.diag(noise_matrix))

    priv_xx      = np.matmul(features.T, features) + sensitivity * (bound_x**2) * noise_matrix

    # Line 5 in Algorithm 2
    priv_xy      = np.dot(features.T, labels).flatten() + sensitivity * bound_x * bound_y * np.random.normal(size=d)

    # Output of Algorithm 2
    model_adassp = np.matmul(np.linalg.pinv(priv_xx + final_lambda * np.eye(d)), priv_xy)

    return model_adassp


class AdasspBasedDPLinearRegression(DifferentiallyPrivateLinearRegression):

    def __init__(self,
                 privacy_notion,
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
        [1] Wang, Yu-Xiang. “Revisiting differentially private linear regression: optimal and adaptive prediction &
        estimation in unbounded domain.” Conference on Uncertainty in Artificial Intelligence (2018).
        """

        super().__init__(privacy_notion = privacy_notion,
                         accountant     = accountant)

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

        self.coef_   = adassp(features = X,
                              labels   = y,
                              epsilon  = self.epsilon,
                              delta    = self.delta)

        if y.ndim == 1:
            self.coef_ = np.ravel(self.coef_)

        self._set_intercept(None, None, None)

        # Deduct the privacy cost from the privacy budget
        # -------------------------------------------------------------------------------------------------------------
        self.accountant.spend(self.epsilon, self.delta)

        return self
