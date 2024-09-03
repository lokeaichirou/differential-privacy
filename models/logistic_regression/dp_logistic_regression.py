
from sklearn import linear_model
from sklearn.utils.multiclass import check_classification_targets

from accountant import BudgetAccountant
from utils import PrivacyLeakWarning, warn_unused_args, check_random_state
from bounds import DiffprivlibMixin
from tailed_notions import *


class DifferentiallyPrivateLogisticRegression(linear_model.LogisticRegression, DiffprivlibMixin):
    _parameter_constraints = DiffprivlibMixin._copy_parameter_constraints(linear_model.LogisticRegression,
                                                                          "tol",
                                                                          "C",
                                                                          "fit_intercept",
                                                                          "max_iter",
                                                                          "verbose",
                                                                          "warm_start",
                                                                          "n_jobs",
                                                                          "random_state")

    def __init__(self,
                 *,
                 privacy_notion,
                 tol           = 1e-4,
                 C             = 1.0,
                 fit_intercept = True,
                 max_iter      = 100,
                 verbose       = 0,
                 n_jobs        = None,
                 random_state  = None,
                 accountant    = None,
                 **unused_args):
        """

        Logistic Regression with differential privacy base class.

        :param privacy_notion:
        :param epsilon:       float type.                Privacy parameter ${epsilon}.
        :param tol:           float type, default: 1e-4  Tolerance for stopping criteria.

        :param C:             float type, default: 1.0   Inverse of regularization strength; must be a positive float.
                                                         Like in support vector machines, smaller values specify
                                                         stronger regularization.

        :param fit_intercept: bool, default: True        Specifies if a constant (a.k.a. bias or intercept) should be
                                                         added to the decision function.

        :param max_iter:      int type, default: 100     Maximum number of iterations taken for the solver to converge.
                                                         For smaller `epsilon` (more noise), `max_iter` may need to be
                                                         increased.
        :param verbose:       int type, default: 0       Set to any positive number for verbosity.
        :param n_jobs:        int, optional              Number of CPU cores used when parallelizing over classes. None
                                                         means 1 unless in a context. -1 means using all processors.

        :param random_state:  int or RandomState, optional. Controls the randomness of the model. To obtain a
                                                            deterministic behaviour during randomisation, random_state
                                                            has to be fixed to an integer.

        :param accountant:    BudgetAccountant, optional. Accountant to keep track of privacy budget.
        :param unused_args:

        """

        super().__init__(tol           = tol,
                         C             = C,
                         fit_intercept = fit_intercept,
                         random_state  = random_state,
                         max_iter      = max_iter,
                         multi_class   = 'ovr',
                         verbose       = verbose,
                         n_jobs        = n_jobs)

        self.privacy_notion = privacy_notion
        self.classes        = None
        self.accountant     = BudgetAccountant.load_default(accountant)

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

    def necessary_steps(self, X, y, sample_weight=None):

        self.parse_privacy_notion()

        # Validate
        # -------------------------------------------------------------------------------------------------------------
        self.accountant.check(self.epsilon, self.delta)

        if sample_weight is not None:
            self._warn_unused_args("sample_weight")

        random_state = check_random_state(self.random_state)

        # Validate and preprocess training data
        # -------------------------------------------------------------------------------------------------------------
        X, y = self._validate_data(X,
                                   y,
                                   accept_sparse       = 'csr',
                                   dtype               = float,
                                   order               = "C",
                                   accept_large_sparse = True)

        # Check classification target and count number of classes
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        return random_state
