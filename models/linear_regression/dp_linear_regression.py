
import warnings
import sklearn.linear_model as sk_lr
from sklearn.utils import check_consistent_length

from utils import PrivacyLeakWarning, check_random_state
from bounds import DiffprivlibMixin
from accountant import BudgetAccountant
from tailed_notions import *


class DifferentiallyPrivateLinearRegression(sk_lr.LinearRegression, DiffprivlibMixin):

    _parameter_constraints = DiffprivlibMixin._copy_parameter_constraints(sk_lr.LinearRegression,
                                                                          "fit_intercept",
                                                                          "copy_X")

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
        Linear Regression with differential privacy base class.

        :param privacy_notion:
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
        """

        super().__init__(fit_intercept = fit_intercept, copy_X = copy_X, n_jobs = None)

        self.privacy_notion         = privacy_notion
        self.data_processing_method = data_processing_method
        self.bounds_X               = bounds_X
        self.bounds_y               = bounds_y
        self.random_state           = random_state
        self.accountant             = BudgetAccountant.load_default(accountant)

        self._warn_unused_args(unused_args)

    def parse_privacy_notion(self):
        # Parse privacy_notion
        if isinstance(self.privacy_notion, EpsilonDp):
            self.epsilon = self.privacy_notion.epsilon
            self.delta   = 0

        elif isinstance(self.privacy_notion, EpsilonDeltaDp):
            self.epsilon = self.privacy_notion.epsilon
            self.delta   = self.privacy_notion.delta

        elif isinstance(self.privacy_notion, RenyiDp):
            self.alpha       = self.privacy_notion.alpha
            self.epsilon_bar = self.privacy_notion.epsilon_bar
            self.delta       = self.privacy_notion.delta
            self.epsilon     = self.epsilon_bar + np.log(1 / self.delta) / (self.alpha - 1)

    def necessary_steps(self, X, y, sample_weight):
        # Parse privacy notion
        # -------------------------------------------------------------------------------------------------------------
        self.parse_privacy_notion()

        # Validate
        # -------------------------------------------------------------------------------------------------------------
        self.accountant.check(self.epsilon, self.delta)

        if sample_weight is not None:
            self._warn_unused_args("sample_weight")

        random_state = check_random_state(self.random_state)

        # Validate training data
        # -------------------------------------------------------------------------------------------------------------
        X, y = self._validate_data(X,
                                   y,
                                   accept_sparse=False,
                                   y_numeric=True,
                                   multi_output=True)

        check_consistent_length(X, y)

        # Check bounds
        # -------------------------------------------------------------------------------------------------------------
        if self.bounds_X is None or self.bounds_y is None:
            warnings.warn("Bounds parameters haven't been specified, so falling back to determining bounds from the "
                          "data.\n"
                          "This will result in additional privacy leakage. To ensure differential privacy with no "
                          "additional privacy loss, specify `bounds_X` and `bounds_y`.",
                          PrivacyLeakWarning)

            if self.bounds_X is None:
                self.bounds_X = (np.min(X, axis=0), np.max(X, axis=0))
            if self.bounds_y is None:
                self.bounds_y = (np.min(y, axis=0), np.max(y, axis=0))

        self.bounds_X = self._check_bounds(self.bounds_X, X.shape[1])
        self.bounds_y = self._check_bounds(self.bounds_y, y.shape[1] if y.ndim > 1 else 1)

        # Get the number of features and targets from data, and compute epsilon_intercept_scale
        # -------------------------------------------------------------------------------------------------------------
        self.n_features = X.shape[1]
        self.n_targets  = y.shape[1] if y.ndim > 1 else 1

        return random_state
