
import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES

from utils import warn_unused_args, check_random_state
from bounds import check_bounds, clip_to_bounds
from statistical_tools.mean import mean
from accountant import BudgetAccountant


def standardize_data(X,
                     fit_intercept  = True,
                     epsilon        = 1.0,
                     bounds_X       = None,
                     copy           = True,
                     check_input    = True,
                     random_state   = None,
                     use_noisy_mean = False,
                     **unused_args):
    """
    Standardize the data of both input X and output y by minus their mean values,

    :param X:
    :param y:
    :param fit_intercept:
    :param epsilon:
    :param bounds_X:
    :param bounds_y:
    :param copy:
    :param check_input:
    :param random_state:
    :param unused_args:
    :return:
    """
    warn_unused_args(unused_args)
    random_state = check_random_state(random_state)

    if check_input:
        X = check_array(X, copy=copy, accept_sparse=False, dtype=FLOAT_DTYPES)
    elif copy:
        X = X.copy(order='K')

    if fit_intercept:
        bounds_X = check_bounds(bounds_X, X.shape[1] if X.ndim > 1 else 1)

        X = clip_to_bounds(X, bounds_X)

        X_offset = mean(X,
                        axis           = 0,
                        bounds         = bounds_X,
                        epsilon        = epsilon,
                        random_state   = random_state,
                        accountant     = BudgetAccountant(),
                        use_noisy_mean = use_noisy_mean)
        X -= X_offset

    else:
        if X.ndim == 1:
            X_offset = X.dtype.type(0)
        else:
            X_offset = np.zeros(X.shape[1], dtype=X.dtype)

    return X, X_offset

