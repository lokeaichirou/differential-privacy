from numpy.core import multiarray as mu
from numpy.core import umath as um

from ..statistical_tools.variance import variance


def standard_deviation(array,
                       privacy_notion,
                       bounds          = None,
                       axis            = None,
                       dtype           = None,
                       keepdims        = False,
                       random_state    = None,
                       single_querying = True,
                       accountant      = None,
                       nan             = False):

    v = variance(array,
                 privacy_notion  = privacy_notion,
                 bounds          = bounds,
                 axis            = axis,
                 dtype           = dtype,
                 keepdims        = keepdims,
                 random_state    = random_state,
                 single_querying = single_querying,
                 accountant      = accountant,
                 nan             = nan)

    if isinstance(v, mu.ndarray):
        output = um.sqrt(v)
    elif hasattr(v, 'dtype'):
        output = v.dtype.type(um.sqrt(v))
    else:
        output = um.sqrt(v)

    return output
