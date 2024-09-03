import numpy as np

from ..accountant import BudgetAccountant
from ..statistical_tools.statistical_utils import _wrap_axis, check
from ..bounds import check_bounds, clip_to_bounds
from ..mechanisms.laplace import LaplaceTruncated
from ..mechanisms.gaussian import Gaussian
from ..mechanisms.renyi import Renyi
from ..tailed_notions import *


def dp_sum(array,
        privacy_notion,
        bounds       = None,
        axis         = None,
        dtype        = None,
        keepdims     = False,
        random_state = None,
        accountant   = None,
        nan          = False,
        **unused_args):

    """
    Sum of array elements over a given axis with differential privacy.

    :param array: array_like;                    Array containing numbers whose mean is to be computed. If array is not
                                                 an array, a conversion is performed.
    :param epsilon: float type, default: 1.0;    Privacy parameter ${epsilon}
    :param bounds: tuple type, optional;         Bounds of the values of the array, which is in the form of (min, max).
    :param axis: int or tuple of ints, optional; Axis or axes along which the means are computed.
    :param dtype: data-type, optional;           Type to use in computing the mean.
    :param keepdims:bool, default: False;        If set to be True, the axes which are reduced are left in the result as
                                                 dimensions with size one. With this option, the result will broadcast
                                                 correctly against the input array.
    :param random_state: int or RandomState, optional; Controls the randomness of the algorithm.
    :param accountant: BudgetAccountant, optional; Accountant to keep track of privacy budget.
    :param nan:
    :param unused_args:
    """

    # Check unused arguments, random_state and whether bounds are provided
    # -------------------------------------------------------------------------------------------------------------
    bounds, random_state = check(array        = array,
                                 unused_args  = unused_args,
                                 random_state = random_state,
                                 bounds       = bounds)

    # Parse privacy_notion
    if isinstance(privacy_notion, EpsilonDp):
        epsilon = privacy_notion.epsilon
        delta = 0

    elif isinstance(privacy_notion, EpsilonDeltaDp):
        epsilon = privacy_notion.epsilon
        delta = privacy_notion.delta

    elif isinstance(privacy_notion, RenyiDp):
        alpha = privacy_notion.alpha
        epsilon_bar = privacy_notion.epsilon_bar
        delta = privacy_notion.delta
        epsilon = epsilon_bar + np.log(1 / delta) / (alpha - 1)

    # Function only needs to be evaluated on scalar outputs.
    # -------------------------------------------------------------------------------------------------------------
    if axis is not None or keepdims:
        return _wrap_axis(sum,
                          array,
                          privacy_notion = privacy_notion,
                          bounds         = bounds,
                          axis           = axis,
                          dtype          = dtype,
                          keepdims       = keepdims,
                          random_state   = random_state,
                          accountant     = accountant,
                          nan            = nan)

    # Check whether specified bounds are valid or not
    # -------------------------------------------------------------------------------------------------------------
    lower, upper = check_bounds(bounds, shape=0, dtype=dtype)

    # Privacy accountant
    # -------------------------------------------------------------------------------------------------------------
    accountant = BudgetAccountant.load_default(accountant)
    accountant.check(epsilon, delta)

    # Ravel array to be single-dimensional
    # -------------------------------------------------------------------------------------------------------------
    array = clip_to_bounds(np.ravel(array), bounds)

    _func = np.nansum if nan else np.sum
    actual_sum = _func(array, axis=axis, dtype=dtype, keepdims=keepdims)

    if isinstance(privacy_notion, EpsilonDp):
        mech = LaplaceTruncated(epsilon      = epsilon,
                                delta        = 0,
                                sensitivity  = upper - lower,
                                lower        = lower,
                                upper        = upper,
                                random_state = random_state)

    elif isinstance(privacy_notion, EpsilonDeltaDp):
        mech = Gaussian(epsilon      = epsilon,
                        delta        = delta,
                        sensitivity  = upper - lower,
                        random_state = random_state)

    elif isinstance(privacy_notion, RenyiDp):
        mech = Renyi(alpha        = alpha,
                     epsilon_bar  = epsilon_bar,
                     delta        = delta,
                     sensitivity  = upper - lower,
                     random_state = random_state)

    output = mech.randomise(actual_sum)
    accountant.spend(epsilon, delta)

    return output
