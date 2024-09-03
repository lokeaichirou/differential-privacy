
"""
Support Vector Machine with differential privacy by objective function perturbation
"""

import warnings

import numpy as np
from joblib import delayed, Parallel
from scipy import optimize
from scipy.optimize import minimize

from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array, check_consistent_length

from models.support_vector_machine.dp_support_vector_machine import DifferentiallyPrivateSupportVectorMachine
from accountant import BudgetAccountant
from utils import PrivacyLeakWarning, warn_unused_args, check_random_state


class ObjectiveFunctionPerturbationBasedDPSVM(DifferentiallyPrivateSupportVectorMachine):

    def __init__(self,
                 *,
                 privacy_notion,
                 tol           = 1e-4,
                 C             = 10,
                 fit_intercept = True,
                 max_iter      = 100,
                 verbose       = 0,
                 n_jobs        = None,
                 random_state  = None,
                 accountant    = None,
                 **unused_args):
        """

        Implements regularised support vector machine and is optimized by L-BFGS-B algorithm. ${epsilon}-Differential
        privacy is achieved by adding random vector to the cost function of the support vector machine, which is proposed
        by [1].
        This class is a child of DifferentiallyPrivateSupportVectorMachine, with amendments to allow for the implementation
        of differential privacy.

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

        Reference:
        [1]

        """

        super().__init__(privacy_notion    = privacy_notion,
                         tol               = tol,
                         C                 = C,
                         fit_intercept     = fit_intercept,
                         random_state      = random_state,
                         max_iter          = max_iter,
                         multi_class       = 'ovr',
                         verbose           = verbose)

        self.n_jobs = n_jobs
        self.accountant = BudgetAccountant.load_default(accountant)

        self._warn_unused_args(unused_args)

    def fit(self, X, y, sample_weight = None):

        random_state = self.necessary_steps(X, y, sample_weight)

        num_classes               = len(self.classes_)
        classes_                  = self.classes_
        num_samples, num_features = X.shape

        if len(self.classes_) == 2:
            num_classes = 1
            classes_    = classes_[1:]

        # Perform dp-svm by calling dp_svm_handler
        # -------------------------------------------------------------------------------------------------------------

        # In case of the multi-class Classification, we can perform the "one vs rest" strategy, i.e. splits a
        # multi-class classification into one binary classification problem per class, each class classification is to
        # be performed in the way that this certain class is compared against other classes
        path_func = delayed(dp_svm_handler)
        set_of_coefs_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer='processes') \
                        (path_func(X,
                                   y,
                                   epsilon       = self.epsilon / num_classes,
                                   pos_class     = class_,
                                   num_classes   = num_classes,
                                   num_samples   = num_samples,
                                   num_features  = num_features,
                                   Cs            = [self.C],
                                   fit_intercept = self.fit_intercept,
                                   max_iter      = self.max_iter,
                                   tol           = self.tol,
                                   verbose       = self.verbose,
                                   random_state  = random_state,
                                   check_input   = True) for class_ in classes_)

        set_of_coefs_, _, n_iter_ = zip(*set_of_coefs_)
        self.n_iter_              = np.asarray(n_iter_, dtype=np.int32)[:, 0]

        self.coef_ = np.asarray(set_of_coefs_)
        self.coef_ = self.coef_.reshape(num_classes, num_features + int(self.fit_intercept))

        if self.fit_intercept:
            self.intercept_ = self.coef_[:, -1]
            self.coef_      = self.coef_[:, :-1]

        self.accountant.spend(self.epsilon, self.delta)

        return self


def dp_svm_handler(X,
                   y,
                   epsilon,
                   pos_class     = None,
                   Cs            = None,
                   fit_intercept = True,
                   huberconst    = 0.5,
                   max_iter      = 100,
                   tol           = 1e-4,
                   verbose       = 0,
                   check_input   = True,
                   random_state  = None,
                   **unused_args):
    """

    Perform the DP-svm for binary classification.

    :param X:             array-like and sparse matrix, of shape (n_samples, n_features)  Training vector, where
                                                                                          n_samples is the number
                                                                                          of samples and n_features
                                                                                          is the number of features.

    :param y:             array-like, of shape (n_samples,)                               Target vector relative to X.
    :param epsilon:       float type.                                                     Split privacy parameter
                                                                                          ${epsilon}.
    :param pos_class:     float type.                                                     The "one" class to be
                                                                                          classified in this individual
                                                                                          model

    :param Cs:            list type.                                                      Inverse of regularization
                                                                                          strength; must be a positive
                                                                                          float. Like in support vector
                                                                                          machines, smaller values
                                                                                          specify stronger regularization.

    :param fit_intercept: bool, default: True                                             Specifies if a constant
                                                                                          (a.k.a. bias or intercept)
                                                                                          should be added to the
                                                                                          decision function.

    :param max_iter:      int type, default: 100                                          Maximum number of iterations
                                                                                          taken for the solver to
                                                                                          converge. For smaller 'epsilon'
                                                                                          (more noise), 'max_iter' may
                                                                                          need to be increased.
    :param tol:           float type, default: 1e-4                                       Tolerance for stopping criteria.
    :param verbose:       int type, default: 0                                            Set to any positive number for
                                                                                          verbosity.
    :param random_state:
    :param check_input:
    :param unused_args:
    :return:
    """

    warn_unused_args(unused_args)

    # Pre-processing.
    # -------------------------------------------------------------------------------------------------------------
    if check_input:
        X = check_array(X, accept_sparse='csr', dtype=np.float64, accept_large_sparse=True)
        y = check_array(y, ensure_2d=False, dtype=None)
        check_consistent_length(X, y)

    num_samples, original_num_features = X.shape
    num_features                       = original_num_features + int(fit_intercept)
    classes                            = np.unique(y)

    if fit_intercept:
        X = np.insert(X, num_features-1, 1, axis=1)

    if pos_class is None:
        if classes.size > 2:
            raise ValueError('To fit OvR, use the pos_class argument')
        # np.unique(y) gives labels in sorted order.
        pos_class = classes[1]

    # To perform the ovr, we need to mask the labels first.
    # -------------------------------------------------------------------------------------------------------------

    # Mark the non-positive classes if there are more than 2 classes to be classified, so that the binary classification
    # is performed for each class
    masked_y      = get_intermediate_data_info(y,
                                               pos_class,
                                               data_type = X.dtype)

    XY            = X * masked_y[:, np.newaxis]
    output_vec    = np.zeros(num_features, dtype = X.dtype)
    coefs         = []
    n_iter        = np.zeros(len(Cs), dtype=np.int32)

    c         = 1 / (2 * huberconst)  # value for svm
    tmp       = c / (num_samples * 1/Cs[0])
    epsilon_p = epsilon - np.log(1.0 + 2 * tmp + tmp * tmp)

    #if epsilon_p < 1e-4:
    #    raise Exception("Error: Cannot run algorithm" + "for this n_sample, lambda, epsilon and huberconst value")

    # Express the perturbed objective function
    for i, C in enumerate(Cs):

        if epsilon_p < 1e-4:
            vector_b = generate_random_vector(epsilon, num_features, random_state)
        else:
            vector_b = generate_random_vector(epsilon_p, num_features, random_state)

        args                = (XY, num_samples, 1./C, vector_b, huberconst)
        iprint              = [-1, 50, 1, 100, 101][np.searchsorted(np.array([0, 1, 2, 3]), verbose)]

        #output_vec, _, info = optimize.fmin_l_bfgs_b(func    = eval_svm,
        #                                             x0      = output_vec,
        #                                             fprime  = None,
        #                                             args    = args,
        #                                             iprint  = iprint,
        #                                             pgtol   = tol,
        #                                             maxiter = max_iter)

        res = minimize(eval_svm,
                       output_vec,
                       args=args,
                       method="L-BFGS-B",
                       bounds=None,
                       options={'maxfun': np.inf})

        if not res.success:
            raise Exception(res.message)

        w_priv = res.x

        #if info["warnflag"] == 1:
        #    warnings.warn("lbfgs failed to converge. Increase the number of iterations.", ConvergenceWarning)

        # coefs.append(output_vec.copy())
        coefs.append(w_priv.copy())

        # n_iter[i] = info['nit']

    return np.array(coefs), np.array(Cs), n_iter


def huberloss(z, huberconst):
    """Returns normal Huber loss (float) for a sample.

    Args:
        z (float): x_i (1, n_feature) * y_i (int: -/+1) * w (n_feature, 1).
        huberconst (float): Huber loss parameter.

    References:
        chaudhuri2011differentially: equation 7 & corollary 13
    """
    if z > 1.0 + huberconst:
        hloss = 0
    elif z < 1.0 - huberconst:
        hloss = 1 - z
    else:
        hloss = (1 + huberconst - z) ** 2 / (4 * huberconst)
    return hloss


def eval_svm(weights, XY, num, lambda_, b, huberconst):
    """
    Evaluates differentially private regularized svm loss for a data set.

    :param weights:    weights (ndarray of shape (n_feature,)): Weights in w'x in classifier.
    :param XY:         (ndarray of shape (n_sample, n_feature)): each row x_i (1, n_feature) * label y_i (int: -/+1).
    :param num:        (int): n_sample.
    :param lambda_:    (float): Regularization parameter.
    :param b:          (ndarray of shape (n_feature,)): Noise vector. If b is a zero vector, returns non-private regularized svm loss.
    :param huberconst: (float): Huber loss parameter.

    :return           fw (float): Differentially private regularized svm loss.
    """

    # add Huber loss from all samples
    XYW = np.matmul(XY, weights)
    fw  = np.mean([huberloss(z=z, huberconst=huberconst) for z in XYW], dtype=np.float64)

    # add regularization term (1/2 * lambda * |w|^2) and b term (1/n * b'w)
    fw += 0.5 * lambda_ * weights.dot(weights) + 1.0 / num * b.dot(weights)
    return fw


def get_intermediate_data_info(y, pos_class, data_type):
    """
    Get some intermediate data info masked y

    :param y:
    :param pos_class:
    :param fit_intercept:
    :return:
    """
    mask         = (y == pos_class)
    y_bin        = np.ones(y.shape, dtype=data_type)
    y_bin[~mask] = -1.0

    return y_bin


def generate_random_vector(epsilon, num_features, random_state):
    """
    Generate_random_vector

    :param epsilon:
    :param num_features:
    :param random_state:
    :return:
    """
    norm_b                 = random_state.gamma(num_features, 2 / epsilon)

    direction_b            = np.random.uniform(low=-1, high=1, size=num_features)
    norm_direction_b       = np.linalg.norm(direction_b, 2)
    normalized_direction_b = direction_b / norm_direction_b

    return normalized_direction_b * norm_b
