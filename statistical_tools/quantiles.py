import numpy as np

from ..accountant import BudgetAccountant
from ..statistical_tools.statistical_utils import _wrap_axis, check
from ..bounds import check_bounds, clip_to_bounds
from ..mechanisms.exponential import Exponential
import copy


def quantile(array,
             quant,
             privacy_notion,
             bounds          = None,
             axis            = None,
             keepdims        = False,
             random_state    = None,
             single_querying = True,
             accountant      = None,
             **unused_args):
    """
    Compute the differentially private quantile of the array.

    Returns the specified quantile with differential privacy.  The quantile is calculated over the flattened array.

    :param array:    array_like;                       Array containing numbers whose mean is to be computed. If array
                                                       is not an array, a conversion is performed.
    :param quant:    float or array-like.              Quantile or array of quantiles. Each quantile must be in the unit
                                                       interval [0, 1]. If quant is array-like, quantiles are returned
                                                       over the flattened array.
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

    :return: m : ndarray.   Returns a new array containing the quantile values.

    Reference
    # -------------------------------------------------------------------------------------------------------------
    https://dl.acm.org/doi/pdf/10.1145/1993636.1993743
    """

    # Check unused arguments, random_state and whether bounds are provided
    # -------------------------------------------------------------------------------------------------------------
    bounds, random_state = check(array        = array,
                                 unused_args  = unused_args,
                                 random_state = random_state,
                                 bounds       = bounds)

    quant = np.ravel(quant)

    if np.any(quant < 0) or np.any(quant > 1):
        raise ValueError("Quantiles must be in the unit interval [0, 1].")

    if len(quant) > 1:
        new_privacy_notion         = copy.deepcopy(privacy_notion)
        new_privacy_notion.epsilon = new_privacy_notion.epsilon / len(quant)

        return np.array([quantile(array,
                                  q_i,
                                  privacy_notion = new_privacy_notion,
                                  bounds         = bounds,
                                  axis           = axis,
                                  keepdims       = keepdims,
                                  accountant     = accountant) for q_i in quant])

    # Dealing with a single quant from now on
    quant = quant.item()

    # Function only needs to be evaluated on scalar outputs.
    if axis is not None or keepdims:
        return _wrap_axis(quantile,
                          array,
                          quant          = quant,
                          privacy_notion = privacy_notion,
                          bounds         = bounds,
                          axis           = axis,
                          keepdims       = keepdims,
                          random_state   = random_state,
                          accountant     = accountant)

    # Check whether specified bounds are valid or not
    # -------------------------------------------------------------------------------------------------------------
    bounds = check_bounds(bounds, shape=0, min_separation=1e-5)

    # Privacy accountant
    # -------------------------------------------------------------------------------------------------------------
    if single_querying:
        accountant = BudgetAccountant.load_default(accountant)
        accountant.check(privacy_notion.epsilon, 0)

    # Ravel array to be single-dimensional
    # -------------------------------------------------------------------------------------------------------------
    array = clip_to_bounds(np.ravel(array), bounds)

    k     = array.size
    array = np.append(array, list(bounds))
    array.sort()

    interval_sizes = np.diff(array)

    if np.isnan(interval_sizes).any():
        return np.nan

    mech   = Exponential(epsilon      = privacy_notion.epsilon,
                         sensitivity  = 1,
                         utility      = list(-np.abs(np.arange(0, k + 1) - quant * k)),
                         measure      = list(interval_sizes),
                         random_state = random_state)

    idx    = mech.randomise()
    output = random_state.random() * (array[idx+1] - array[idx]) + array[idx]

    if single_querying:
        accountant.spend(privacy_notion.epsilon, 0)

    return output
