import numpy as np
import warnings
from .. import utils
from sys import maxsize

from ..accountant import BudgetAccountant
from ..statistical_tools.statistical_utils import check
from ..mechanisms.laplace import LaplaceTruncated
from ..mechanisms.gaussian import Gaussian
from ..mechanisms.renyi import Renyi
from ..tailed_notions import *


def histogram(array,
              privacy_notion,
              bounds       = None,
              bins         = 10,
              range        = None,
              weights      = None,
              density      = None,
              random_state = None,
              accountant   = None,
              dtype        = None,
              **unused_args):
    """
    Compute the differentially private histogram of a set of data.

    :param array:        array_like. Input data.       The histogram is computed over the flattened array.
    :param epsilon:      float, default: 1.0.          Privacy parameter ${epsilon}.
    :param bins:         int or sequence of scalars or str, default: 10.
    :param range:        (float, float), optional.
    :param weights:      array_like, optional.
    :param density:      bool, optional
    :param random_state: int or RandomState, optional. Controls the randomness of the algorithm. To obtain a
                                                       deterministic behaviour during randomisation, random_state has to
                                                       be fixed to an integer.
    :param accountant:   BudgetAccountant, optional.   Accountant to keep track of privacy budget.
    :param unused_args:
    :return:

    """
    # Check unused arguments, random_state and whether bounds are provided
    # -------------------------------------------------------------------------------------------------------------
    bounds, random_state = check(array=array,
                                 unused_args=unused_args,
                                 random_state=random_state,
                                 bounds=bounds)

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

    # Privacy accountant
    # -------------------------------------------------------------------------------------------------------------
    accountant = BudgetAccountant.load_default(accountant)
    accountant.check(epsilon, delta)

    if range is None:
        warnings.warn("Range parameter has not been specified. Falling back to taking range from the data.\n"
                      "To ensure differential privacy, and no additional privacy leakage, the range must be "
                      "specified independently of the data (i.e., using domain knowledge).", utils.PrivacyLeakWarning)

    # Add LaplaceTruncated noise to the actual histogram
    # -------------------------------------------------------------------------------------------------------------
    hist, bin_edges = np.histogram(array,
                                   bins    = bins,
                                   range   = range,
                                   weights = weights,
                                   density = None)

    if isinstance(privacy_notion, EpsilonDp):
        dp_mech = LaplaceTruncated(epsilon      = epsilon,
                                   sensitivity  = 2,
                                   lower        = 0,
                                   upper        = maxsize,
                                   random_state = random_state)

    elif isinstance(privacy_notion, EpsilonDeltaDp):
        dp_mech = Gaussian(epsilon      = epsilon,
                           delta        = delta,
                           sensitivity  = 2,
                           random_state = random_state)

    elif isinstance(privacy_notion, RenyiDp):
        dp_mech = Renyi(alpha        = alpha,
                        epsilon_bar  = epsilon_bar,
                        delta        = delta,
                        sensitivity  = 2,
                        random_state = random_state)

    dp_hist         = np.zeros_like(hist)
    for i in np.arange(dp_hist.shape[0]):
        dp_hist[i] = dp_mech.randomise(int(hist[i]))

    accountant.spend(epsilon, delta)

    if density:
        bin_sizes = np.array(np.diff(bin_edges), float)
        return dp_hist / bin_sizes / (dp_hist.sum() if dp_hist.sum() else 1), bin_edges

    return dp_hist, bin_edges
