
"""
General utilities and tools developed to assist the differentially private operations on data.
"""

import numpy as np
import warnings
from bounds import check_bounds
import utils

import copy
from tailed_notions import EpsilonDp


def _wrap_axis(func,
               array,
               *,
               axis,
               keepdims,
               privacy_notion,
               bounds,
               **kwargs):
    """
    Wrapper for functions with axis and keepdims parameters to ensure the function only needs to be evaluated on scalar
    outputs.

    """

    dummy  = np.zeros_like(array).sum(axis=axis, keepdims=keepdims)
    array  = np.asarray(array)
    ndim   = array.ndim
    bounds = check_bounds(bounds, np.size(dummy) if np.ndim(dummy) == 1 else 0)

    if isinstance(axis, int):
        axis = (axis,)
    elif axis is None:
        axis = tuple(range(ndim))

    # Ensure all axes are non-negative
    axis = tuple(ndim + ax if ax < 0 else ax for ax in axis)

    if isinstance(dummy, np.ndarray):
        iterator                   = np.nditer(dummy, flags=['multi_index'])
        if not isinstance(privacy_notion, EpsilonDp):
            new_privacy_notion         = None
        else:
            new_privacy_notion         = copy.deepcopy(privacy_notion)
            new_privacy_notion.epsilon = privacy_notion.epsilon / dummy.size

        while not iterator.finished:
            idx     = list(iterator.multi_index)  # Multi index on 'dummy'
            _bounds = (bounds[0][idx], bounds[1][idx]) if np.ndim(dummy) == 1 else bounds

            # Construct slicing tuple on array
            if len(idx) + len(axis) > ndim:
                full_slice = tuple(slice(None) if ax in axis else idx[ax] for ax in range(ndim))
            else:
                idx.reverse()
                full_slice = tuple(slice(None) if ax in axis else idx.pop() for ax in range(ndim))

            dummy[iterator.multi_index] = func(array[full_slice],
                                               privacy_notion = new_privacy_notion,
                                               bounds         = _bounds,
                                               **kwargs)
            iterator.iternext()

        return dummy

    return func(array, bounds=bounds, privacy_notion = privacy_notion, **kwargs)


def check(array, unused_args, random_state, bounds):
    """
    Check unused arguments, random_state and whether bounds are provided

    :param unused_args:
    :param random_state:
    :param bounds:
    :return:
    """
    utils.warn_unused_args(unused_args)
    random_state = utils.check_random_state(random_state)

    if bounds is None:
        warnings.warn("Bounds are not specified and will be inferred by the data provided, but may result in privacy "
                      "leakage. To ensure differential privacy and no additional privacy leakage, it is better to"
                      "bounds for each dimension.", utils.PrivacyLeakWarning)
        bounds = (np.min(array), np.max(array))

    return bounds, random_state
