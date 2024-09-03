"""
Linear Regression with differential privacy
"""

from scipy.optimize import minimize

from models.linear_regression.dp_linear_regression import DifferentiallyPrivateLinearRegression

# from dp_linear_regression import DifferentiallyPrivateLinearRegression
from sensitivity_computation.compute_sensitivity_based_on_data import compute_up_to_second_order_polynomial_based_sensitivity
from mechanisms.laplace import Laplace, LaplaceFolded
from mechanisms.gaussian import Gaussian
from mechanisms.renyi import Renyi
from tailed_notions import *


def generate_mech(degree_order,
                  privacy_notion,
                  epsilon,
                  sensitivity,
                  random_state,
                  lower             = None,
                  upper             = None):

    if isinstance(privacy_notion, EpsilonDp):
        if degree_order == 0 or degree_order == 2:
            mech = LaplaceFolded(epsilon      = epsilon,
                                 sensitivity  = sensitivity,
                                 lower        = lower,
                                 upper        = upper,
                                 random_state = random_state)
        elif degree_order == 1:
            mech = Laplace(epsilon      = epsilon,
                           sensitivity  = sensitivity,
                           random_state = random_state)

    elif isinstance(privacy_notion, EpsilonDeltaDp):
        mech = Gaussian(epsilon      = epsilon,
                        delta        = privacy_notion.delta,
                        sensitivity  = sensitivity,
                        random_state = random_state)

    elif isinstance(privacy_notion, RenyiDp):
        mech = Renyi(alpha        = privacy_notion.alpha,
                     epsilon_bar  = privacy_notion.epsilon_bar,
                     delta        = privacy_notion.delta,
                     sensitivity  = sensitivity,
                     random_state = random_state)

    return mech


def _construct_regression_obj(X,
                              y,
                              bounds_X,
                              bounds_y,
                              privacy_notion,
                              epsilon,
                              alpha,
                              random_state,
                              data_processing_method,
                              renyi_alpha = None):
    """
    In the original paper [1], the differential privacy is achieved by adding noise to the coefficients related to X and
    y (mono_coef_0, mono_coef_1, mono_coef_2 in section 4.2 in the paper) in the expansion of the objective function as 
    introduced by Algorithm 1 in the paper. Then the model parameter w, which is omega here will be computed by
    minimizing this expansion of cost function for linear regression in section 4.2 in the paper.
    
    In [1], the global sensitivity is based on the assumption on training data distribution, which is the L2-norm of X
    is less than 1, y is within [-1, 1] and such global sensitivity is equal to a constant of 2*(1+n_features)**2 for
    the case of linear regression model, and this global sensitivity is applied for every order of polynomial terms
    (which are 0-th, 1-st, 2-nd order for the cost function of linear regression) computed based on data, there would
    be larger noise to be added than being required in reality. Hence, we can infer the sensitivity based on the real
    data alternatively and it depends more on the real data property.

    We regress every target variable independently

    :param X:                   array-like or sparse matrix, shape (n_samples, n_features). Training data
    :param y:                   array_like, shape (n_samples, n_targets).                   Target values.
    :param bounds_X:                                                                        bounds of X
    :param bounds_y:                                                                        bounds of y
    :param epsilon:             float type.                                                 Privacy parameter ${epsilon}.
    :param alpha:
    :param random_state:
    :param scale_training_data: bool type.                                                  Whether we use the data being
                                                                                            preprocessed data with 
                                                                                            bounds_X, bounds_y = (0, 1), 
                                                                                            (-1, 1) 

    Reference:
    [1] Zhang, Jun, Zhenjie Zhang, Xiaokui Xiao, Yin Yang, and Marianne Winslett. "Functional mechanism: regression
    analysis under differential privacy." arXiv preprint arXiv:1208.0219 (2012)..

    """
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    n_features, n_targets = X.shape[1], y.shape[1]

    global_epsilon = epsilon
    local_epsilon  = epsilon / (1 + n_targets * n_features + n_features * (n_features + 1) / 2)
    coefs          = ((y ** 2).sum(axis=0), np.einsum('ij,ik->jk', X, y), np.einsum('ij,ik', X, X))

    del X, y

    # global sensitivity value introduced by [1]
    global_sensitivity = 2*(1+n_features)**2

    # Randomise 0th-degree monomial coefficients
    # -------------------------------------------------------------------------------------------------------------
    mono_coef_0 = np.zeros(n_targets)
    for i in range(n_targets):
        if data_processing_method is None or data_processing_method == 'standardize':
            local_sensitivity = np.abs([bounds_y[0][i], bounds_y[1][i]]).max() ** 2
            mech = generate_mech(degree_order   = 0,
                                 privacy_notion = privacy_notion,
                                 epsilon        = global_epsilon,
                                 sensitivity    = local_sensitivity,
                                 random_state   = random_state,
                                 lower          = 0,
                                 upper          = float("inf"))

        else:
            mech = generate_mech(degree_order   = 0,
                                 privacy_notion = privacy_notion,
                                 epsilon        = global_epsilon,
                                 sensitivity    = global_sensitivity,
                                 random_state   = random_state,
                                 lower          = 0,
                                 upper          = float("inf"))

        mono_coef_0[i] = mech.randomise(coefs[0][i])

    # Randomise 1st-degree monomial coefficients
    # -------------------------------------------------------------------------------------------------------------
    mono_coef_1 = np.zeros((n_features, n_targets))
    for i in range(n_targets):
        for j in range(n_features):
            if data_processing_method is None or data_processing_method == 'standardize':
                local_sensitivity = compute_up_to_second_order_polynomial_based_sensitivity(bounds_y[0][i],
                                                                                            bounds_y[1][i],
                                                                                            bounds_X[0][j],
                                                                                            bounds_X[1][j])
                mech = generate_mech(degree_order   = 1,
                                     privacy_notion = privacy_notion,
                                     epsilon        = global_epsilon,
                                     sensitivity    = local_sensitivity,
                                     random_state   = random_state)

            else:
                mech = generate_mech(degree_order   = 1,
                                     privacy_notion = privacy_notion,
                                     epsilon        = global_epsilon,
                                     sensitivity    = global_sensitivity,
                                     random_state   = random_state)

            mono_coef_1[j, i] = mech.randomise(coefs[1][j, i])

    # Randomise 2nd-degree monomial coefficients
    # -------------------------------------------------------------------------------------------------------------
    mono_coef_2 = np.zeros((n_features, n_features))
    for i in range(n_features):
        if not data_processing_method is None or data_processing_method == 'standardize':
            local_sensitivity = np.max(np.abs([bounds_X[0][i], bounds_X[0][i]])) ** 2

            mech = generate_mech(degree_order   = 2,
                                 privacy_notion = privacy_notion,
                                 epsilon        = global_epsilon,
                                 sensitivity    = local_sensitivity,
                                 random_state   = random_state,
                                 lower          = 0,
                                 upper          = float("inf"))

        else:
            mech = generate_mech(degree_order   = 2,
                                 privacy_notion = privacy_notion,
                                 epsilon        = global_epsilon,
                                 sensitivity    = global_sensitivity,
                                 random_state   = random_state,
                                 lower          = 0,
                                 upper          = float("inf"))

        mono_coef_2[i, i] = mech.randomise(coefs[2][i, i])

        for j in range(i + 1, n_features):
            if data_processing_method is None or data_processing_method == 'standardize':
                local_sensitivity       = compute_up_to_second_order_polynomial_based_sensitivity(bounds_X[0][i],
                                                                                                  bounds_X[1][i],
                                                                                                  bounds_X[0][j],
                                                                                                  bounds_X[1][j])

                mech = generate_mech(degree_order   = 2,
                                     privacy_notion = privacy_notion,
                                     epsilon        = local_epsilon,
                                     sensitivity    = local_sensitivity,
                                     random_state   = random_state,
                                     lower          = 0,
                                     upper          = float("inf"))

            else:
                mech = generate_mech(degree_order   = 2,
                                     privacy_notion = privacy_notion,
                                     epsilon        = global_epsilon,
                                     sensitivity    = global_sensitivity,
                                     random_state   = random_state,
                                     lower          = 0,
                                     upper          = float("inf"))

            mono_coef_2[i, j] = mech.randomise(coefs[2][i, j])
            mono_coef_2[j, i] = mono_coef_2[i, j]  # Enforce symmetry

    del coefs
    noisy_coefs = (mono_coef_0, mono_coef_1, mono_coef_2)

    # objective function to be optimized which is the expansion of cost function for linear regression
    # -------------------------------------------------------------------------------------------------------------
    def obj(idx):
        """
        
        :param idx: target index
        :return: 
        """
        def inner_obj(omega):
            func = noisy_coefs[0][idx]
            func -= 2 * np.dot(noisy_coefs[1][:, idx], omega)
            func += np.multiply(noisy_coefs[2], np.tensordot(omega, omega, axes=0)).sum()
            func += alpha * (omega ** 2).sum()

            grad = - 2 * noisy_coefs[1][:, idx] + 2 * np.matmul(noisy_coefs[2], omega) + 2 * omega * alpha

            return func, grad

        return inner_obj

    # Regress every target variable independently
    output = tuple(obj(i) for i in range(n_targets))

    return output, noisy_coefs


class FunctionalMechanismBasedDPLinearRegression(DifferentiallyPrivateLinearRegression):

    def __init__(self,
                 *,
                 privacy_notion,
                 data_processing_method = None,
                 bounds_X               = None,
                 bounds_y               = None,
                 fit_intercept          = False,
                 copy_X                 = True,
                 random_state           = None,
                 accountant             = None,
                 **unused_args):

        """
        Linear Regression with differential privacy class based on functional mechanism proposed by [1]. LinearRegression
        fits a linear model with coefficients w = (w1, ..., wp) to minimize the residual sum of squares between the
        observed targets in the dataset, and the targets predicted by the linear approximation. Differential privacy is
        achieved by adding noise to the expansion of objective function for linear regression.

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
        :param unused_args:
               coef_:         array of shape (n_features, ) or (n_targets, n_features). Estimated coefficients for the
                                                                                        linear regression problem. If
                                                                                        multiple targets are passed
                                                                                        during the fit (y 2D), this is
                                                                                        a 2D array of shape
                                                                                        (n_targets, n_features), while
                                                                                        if only one target is passed,
                                                                                        this is a 1D array of length
                                                                                        n_features.
               intercept_:    float or array of shape of (n_targets,).  Independent term in the linear model. Set to
                                                                        0.0 if False.

        Reference:
        [1] Zhang, Jun, Zhenjie Zhang, Xiaokui Xiao, Yin Yang, and Marianne Winslett. "Functional mechanism: regression
        analysis under differential privacy." arXiv preprint arXiv:1208.0219 (2012)..
        """

        super().__init__(privacy_notion         = privacy_notion,
                         data_processing_method = data_processing_method,
                         bounds_X               = bounds_X,
                         bounds_y               = bounds_y,
                         fit_intercept          = fit_intercept,
                         copy_X                 = copy_X,
                         random_state           = random_state,
                         accountant             = accountant)

    def fit(self, X, y, sample_weight = None):

        """
        Fit linear model.

        :param X:                   array-like or sparse matrix, shape (n_samples, n_features). Training data
        :param y:                   array_like, shape (n_samples, n_targets).                   Target values.
        :param scale_training_data: whether strictly follow the assumption on training data distribution in [1]
        :param sample_weight:
        :return:                    self : returns an instance of self.
        """
        random_state = self.necessary_steps(X, y, sample_weight)

        if self.fit_intercept and self.data_processing_method == 'standardize':
            epsilon_intercept_scale = 1 / (self.n_features + 1)
        else:
            epsilon_intercept_scale = 0

        X = X.values
        y = y.values

        # Construct regression objective functions
        # -------------------------------------------------------------------------------------------------------------
        objs, obj_coefs = _construct_regression_obj(X,
                                                    y,
                                                    self.bounds_X,
                                                    self.bounds_y,
                                                    privacy_notion         = self.privacy_notion,
                                                    epsilon                = self.epsilon * (1 - epsilon_intercept_scale),
                                                    alpha                  = 0,
                                                    random_state           = random_state,
                                                    data_processing_method = 'original')

        coef            = np.zeros((self.n_features, self.n_targets))

        # Optimize the polynomial coefficients
        # -------------------------------------------------------------------------------------------------------------
        for i, obj in enumerate(objs):
            opt_result = minimize(obj, np.zeros(self.n_features), jac=True)
            coef[:, i] = opt_result.x

        # model parameters = self.coef_ = coef.T
        self.coef_      = coef.T

        # obj_coefs are noisy polynomial terms computed based on data
        self._obj_coefs = obj_coefs

        if y.ndim == 1:
            self.coef_ = np.ravel(self.coef_)

        self._set_intercept(None, None, None)

        # Deduct the privacy cost from the privacy budget
        # -------------------------------------------------------------------------------------------------------------
        self.accountant.spend(self.epsilon, self.delta)

        return self
