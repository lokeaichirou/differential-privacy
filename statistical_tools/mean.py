
import numpy as np

from accountant import BudgetAccountant
from statistical_tools.statistical_utils import _wrap_axis, check
from bounds import check_bounds, clip_to_bounds
from mechanisms.laplace import LaplaceTruncated
from mechanisms.gaussian import Gaussian
from mechanisms.renyi import Renyi
from tailed_notions import *


def mean(array,
         privacy_notion  = None,
         bounds          = None,
         axis            = None,
         dtype           = None,
         keepdims        = False,
         random_state    = None,
         single_querying = True,
         accountant      = None,
         nan             = False,
         **unused_args):
    """
    Compute the differentially private arithmetic mean along the specified axis.

    Returns the differentially private average of the array elements. The average is taken over the flattened array
    at the first, otherwise over the specified axis. Then the Laplace noise is added to satisfy differential privacy,
    where sensitivity is calculated using the bounds.

    :param array:    array_like;                       Array containing numbers whose mean is to be computed. If array
                                                       is not an array, a conversion is performed.
    :param epsilon:  float type, default: 1.0;         Privacy parameter ${epsilon}
    :param bounds:   tuple type, optional;             Bounds of the values of the array, which is in the form of (min, max).
    :param axis:     int or tuple of ints, optional;   Axis or axes along which the means are computed.
    :param dtype:    data-type, optional;              Type to use in computing the mean.
    :param keepdims: bool, default: False;             If set to be True, the axes which are reduced are left in the
                                                       result as dimensions with size one. With this option, the result
                                                       will broadcast correctly against the input array.
    :param random_state: int or RandomState, optional; Controls the randomness of the algorithm.
    :param accountant:   BudgetAccountant, optional;   Accountant to keep track of privacy budget.
    :param nan:          whether ignore NaNs or not
    :param unused_args:

    :return: m : a new array containing the mean values.
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
        delta   = 0

    elif isinstance(privacy_notion, EpsilonDeltaDp):
        epsilon = privacy_notion.epsilon
        delta   = privacy_notion.delta

    elif isinstance(privacy_notion, RenyiDp):
        alpha       = privacy_notion.alpha
        epsilon_bar = privacy_notion.epsilon_bar
        delta       = privacy_notion.delta
        epsilon     = epsilon_bar + np.log(1/delta)/(alpha - 1)

    # Function only needs to be evaluated on scalar outputs.
    # -------------------------------------------------------------------------------------------------------------
    if axis is not None or keepdims:
        return _wrap_axis(mean,
                          array,
                          privacy_notion = privacy_notion,
                          bounds       = bounds,
                          axis         = axis,
                          dtype        = dtype,
                          keepdims     = keepdims,
                          random_state = random_state,
                          accountant   = accountant,
                          nan          = nan)

    # Check whether specified bounds are valid or not
    # -------------------------------------------------------------------------------------------------------------
    lower, upper = check_bounds(bounds, shape=0, dtype=dtype)

    # Privacy accountant
    # -------------------------------------------------------------------------------------------------------------
    if single_querying and privacy_notion is not None:
        accountant   = BudgetAccountant.load_default(accountant)
        accountant.check(epsilon, delta)

    # Ravel array to be single-dimensional
    # -------------------------------------------------------------------------------------------------------------
    array       = clip_to_bounds(np.ravel(array), bounds)

    # Add LaplaceTruncated noise to the actual mean
    # -------------------------------------------------------------------------------------------------------------
    _func       = np.nanmean if nan else np.mean
    actual_mean = _func(array, axis=axis, dtype=dtype, keepdims=keepdims)

    if isinstance(privacy_notion, EpsilonDp):
        mech = LaplaceTruncated(epsilon      = epsilon,
                                delta        = 0,
                                sensitivity  = (upper - lower) / array.size,
                                lower        = lower,
                                upper        = upper,
                                random_state = random_state)

    elif isinstance(privacy_notion, EpsilonDeltaDp):
        mech = Gaussian(epsilon      = epsilon,
                        delta        = delta,
                        sensitivity  = (upper - lower) / array.size,
                        random_state = random_state)

    elif isinstance(privacy_notion, RenyiDp):
        mech = Renyi(alpha        = alpha,
                     epsilon_bar  = epsilon_bar,
                     delta        = delta,
                     sensitivity  = (upper - lower) / array.size,
                     random_state = random_state)

    if privacy_notion is not None:
        output = mech.randomise(actual_mean)

    # Record privacy spend if it is single querying
    if single_querying and privacy_notion is not None:
        accountant.spend(epsilon, delta)

    return output if privacy_notion is not None else actual_mean
