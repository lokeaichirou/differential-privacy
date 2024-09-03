
import counts, quantiles, median, mean, standard_deviation
from ..accountant import BudgetAccountant
import pandas as pd
import copy


def describe_data(array,
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

    :param array:    array_like;                       Array containing numbers whose mean is to be computed. If array
                                                       is not an array, a conversion is performed.
    :param privacy_notion:
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
    :return:
    """
    new_privacy_notion = copy.deepcopy(privacy_notion)
    new_privacy_notion.epsilon = privacy_notion.epsilon / 6

    accountant.check(privacy_notion.epsilon, privacy_notion.delta)

    statistics = {"count":  counts.counts(array, new_privacy_notion, bounds, axis, dtype, keepdims, random_state, single_querying=False, nan=nan),
                  "mean":   mean.mean(array, new_privacy_notion, bounds, axis, dtype, keepdims, random_state, single_querying=False, nan=nan),
                  "std":    standard_deviation.standard_deviation(array, new_privacy_notion, bounds, axis, dtype, keepdims, random_state, single_querying=False, nan=nan),
                  "25%":    quantiles.quantile(array, 0.25, new_privacy_notion, bounds, axis, dtype, keepdims, random_state, single_querying=False, nan=nan),
                  "median": median.median(array, new_privacy_notion, bounds, axis, dtype, keepdims, random_state, single_querying=False, nan=nan),
                  "75%":    quantiles.quantile(array, 0.75, new_privacy_notion, bounds, axis, dtype, keepdims, random_state, single_querying=False, nan=nan)
                }


    accountant = BudgetAccountant.load_default(accountant)
    accountant.spend(privacy_notion.epsilon, privacy_notion.delta)

    df = pd.DataFrame.from_dict(statistics, orient='index', columns=['Value'])
    return df
