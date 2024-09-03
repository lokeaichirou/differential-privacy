
import abc
import math
from numbers import Real, Integral
import numpy as np


# Privacy accountant
# -------------------------------------------------------------------------------------------------------------

def check_epsilon_delta(epsilon, delta, allow_zero=False):
    """
    Checks whether epsilon and delta are valid values for differential privacy. An error would be thrown when failure
    happens, otherwise returns nothing.Both epsilon and delta cannot be simultaneously zero, unless allow_zero is set to
    be True.

    :param epsilon: float type;                    ${Epsilon} for differential privacy. Must be non-negative.
    :param delta: float type;                      ${Delta} for differential privacy. Must be within the unit interval [0, 1].
    :param allow_zero: bool type, default: False.  Allow epsilon and delta both be zero.
    :return:
    """

    if not isinstance(epsilon, Real) or not isinstance(delta, Real):
        raise TypeError("Epsilon and delta must be numeric")

    if epsilon < 0:
        raise ValueError("Epsilon must be non-negative")

    if not 0 <= int(delta) <= 1:
        raise ValueError("Delta must be in [0, 1]")

    if not allow_zero and epsilon + delta == 0:
        raise ValueError("Epsilon and Delta cannot both be zero")


class Budget(tuple):
    """
        A base class of tuple with the form of (epsilon, delta) for privacy budgets and costs.
        The class allows for correct comparison/ordering of privacy budget and costs, ensuring that both epsilon and
        delta satisfy the comparison.

    """
    def __new__(cls, epsilon, delta, min_lambda):
        if epsilon < 0:
            raise ValueError("Epsilon must be non-negative")

        if not 0 <= delta <= 1:
            raise ValueError("Delta must be in [0, 1]")

        return tuple.__new__(cls, (epsilon, delta, min_lambda))

    def __gt__(self, other):
        if self.__ge__(other) and not self.__eq__(other):
            return True
        return False

    def __ge__(self, other):
        if self[0] >= other[0] and self[1] >= other[1]:
            return True
        return False

    def __lt__(self, other):
        if self.__le__(other) and not self.__eq__(other):
            return True
        return False

    def __le__(self, other):
        if self[0] <= other[0] and self[1] <= other[1]:
            return True
        return False


class MomentsAccountant:
    _default = None

    # Initialization
    # -------------------------------------------------------------------------------------------------------------
    def __init__(self,
                 epsilon=float("inf"),
                 delta=1.0,
                 moment_orders=32,
                 verbose=True,
                 spent_budget=None):
        """
        Composition based privacy budget accountant for (ε, δ)-differential privacy.
        This class creates a privacy budget accountant to track privacy spend across queries and other data accesses.
        Once instantiated, the NaivePrivacyAccountantForApproxDP stores each privacy spend and iteratively updates the
        total budget spend, raising an error when the budget ceiling (if specified) is exceeded.
        The accountant can be initialised without any maximum budget, to enable users track the total privacy spend
        of their actions without restriction.

        :param epsilon:      float type, default: infinity. Epsilon budget ceiling of the accountant.
        :param delta:        float, default: 1.0.           Delta budget ceiling of the accountant.
        :param spent_budget: list of tuples of the form (epsilon, delta), optional. List of tuples of pre-existing budget
                                                                                    spends.  Allows for a new accountant
                                                                                    to be initialised with spends
                                                                                    extracted from a previous instance.

        """

        # Check whether the provided budget ceilings are valid
        check_epsilon_delta(epsilon, delta)

        # attributes starting with __ are to be protected
        self.__epsilon = epsilon
        self.__min_epsilon = 0 if epsilon == float("inf") else epsilon * 1e-14
        self.__spent_budget = []
        self.__delta = delta

        self._moment_orders = (moment_orders
                               if isinstance(moment_orders, (list, tuple))
                               else range(1, moment_orders + 1))
        self._max_moment_order = max(self._moment_orders)
        self._log_moments = np.zeros(len(self._moment_orders))
        self._verbose = verbose
        self._E2s = []

        # If spent_budget is given, take it into account
        if spent_budget is not None:
            if not isinstance(spent_budget, list):
                raise TypeError("spent_budget must be a list")

            for _epsilon, _delta in spent_budget:
                self.spend(_epsilon, _delta, None, None)

    # Magic Methods
    # -------------------------------------------------------------------------------------------------------------

    # If with command is used, __enter__ and __exit__ will be called automatically.
    def __enter__(self):
        self.old_default = self.pop_default()
        self.set_default()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pop_default()

        if self.old_default is not None:
            self.old_default.set_default()
        del self.old_default

    def __len__(self):
        return len(self.spent_budget)

    # Methods decorated by @property to protect some attributes starting with __
    # -------------------------------------------------------------------------------------------------------------
    @property
    def spent_budget(self):
        """
        List of tuples of the form (epsilon, delta) of spent privacy budget.
        """
        return self.__spent_budget.copy()

    @property
    def epsilon(self):
        """
        Epsilon privacy ceiling of the accountant.
        """
        return self.__epsilon

    @property
    def delta(self):
        """
        Delta privacy ceiling of the accountant.
        """
        return self.__delta

    @abc.abstractmethod
    def _compute_log_moment(self, sigma, q, moment_order):
        """
        Compute high moment of privacy loss.

        Args:
            sigma: the noise sigma, in the multiples of the sensitivity.
            q: the sampling ratio.
            moment_order: the order of moment.
        Returns:
            log E[exp(moment_order * X)]
        """
        pass

    def accumulate_privacy_spending(self, sigma, q, iters=1, reset=False):
        """
        Accumulate privacy spending.

        In particular, accounts for privacy spending when we assume there are num_examples, and we are releasing the
        vector (sum_{i=1}^{num_examples} x_i) + Normal(0, stddev=l2norm_bound*sigma)
        where l2norm_bound is the maximum l2_norm of each example x_i, and the num_examples have been randomly
        selected out of a pool of total_examples.

        Args:
            sigma: the noise sigma, in the multiples of the sensitivity (that is, if the l2norm sensitivity is k,
            then the caller must have added Gaussian noise with stddev=k*sigma to the result of the query).

            q: sampling probability (batch_size / num_examples).

            iters: number of times that noise is added (e.g., #epochs)

            reset: True -> resets accumulator
        """

        if reset:
            self._log_moments = np.zeros(len(self._moment_orders))

        # the following are useful for computing privacy if all moments are nan/inf
        self.q     = q
        self.sigma = sigma
        self.iters = iters

        for i in range(len(self._log_moments)):
            alpha_i = self._compute_log_moment(sigma, q, self._moment_orders[i])

            # composability (Theorem 2.1)
            alpha = iters * alpha_i

            self._log_moments[i] += alpha

    def _compute_delta(self, log_moments, eps):
        """
        Compute delta for given log_moments and eps.

        Args:
            log_moments: the log moments of privacy loss, in the form of pairs of (moment_order, log_moment)
            eps: the target epsilon.
        Returns:
            delta
            min_lambda: moment that gives the minimum value
        """
        min_delta    = 1.0
        min_lambda   = None
        nanInfMoment = []
        valid        = False

        for moment_order, log_moment in log_moments:

            if math.isinf(log_moment) or math.isnan(log_moment):
                nanInfMoment.append(moment_order)
                continue

            valid = True
            if log_moment < moment_order * eps:
                temp = math.exp(log_moment - moment_order * eps)
                if min_delta > temp:
                    min_delta = temp
                    min_lambda = moment_order

        if self._verbose:
            print("Inf or Nan moment orders: %s\n" % nanInfMoment)

        if not valid:
            # avoid numerical instability (inf) and directly compute delta from formula to compute E2
            # (GaussianMomentsAccountant2) by setting k=1
            if self._verbose:
                print("All moments are inf or Nan")
            if self._verbose:
                print("Estimating privacy given min_lambda=1 from last accumulated sigma")

            min_delta = np.exp(self.iters * (np.log(self.q) + 1.0 / self.sigma**2) - eps)

        return min_delta, min_lambda

    def _compute_eps(self, log_moments, delta):
        min_eps      = float("inf")
        min_lambda   = None
        self._eps    = []
        nanInfMoment = []
        valid        = False

        for moment_order, log_moment in log_moments:
            if math.isinf(log_moment) or math.isnan(log_moment):
                nanInfMoment.append(moment_order)
                self._eps.append(None)
                continue

            valid = True
            temp  = (log_moment - math.log(delta)) / moment_order
            self._eps.append(temp)

            if min_eps > temp:
                min_eps = temp
                min_lambda = moment_order

        if self._verbose:
            print("Inf or Nan moment orders: %s\n" % nanInfMoment)

        if not valid:
            # avoid numerical instability (inf) and directly compute delta
            # from formula to compute E2 (GaussianMomentsAccountant2) by setting k=1
            if self._verbose:
                print("All moments are inf or Nan")
            if self._verbose:
                print("Estimating privacy min_lambda=1 from last accumulated sigma")

            min_eps = self.iters * (np.log(self.q) + 1.0 / self.sigma**2) - np.log(delta)

        return min_eps, min_lambda

    def get_privacy_spent(self, target_eps=None, target_deltas=None):
        """
        Compute privacy spending in (e, d)-DP form for a single or list of eps.

        Args:
            target_eps:    a list of target epsilon's for which we would like to compute corresponding delta value.
            target_deltas: a list of target deltas for which we would like to compute the corresponding eps value.
                           Caller must specify either target_eps or target_delta.
        Returns:
            A list of EpsDelta pairs.
        """
        assert (target_eps is None) ^ (target_deltas is None)
        eps_deltas             = []
        log_moments_with_order = zip(self._moment_orders, self._log_moments)

        if target_eps is not None:
            for eps in target_eps:
                delta, min_lambda = self._compute_delta(log_moments_with_order, eps)
                eps_deltas.append(Budget(eps, delta, min_lambda))
        else:
            assert target_deltas
            for delta in target_deltas:
                eps, min_lambda = self._compute_eps(log_moments_with_order, delta)
                eps_deltas.append(Budget(eps, delta, min_lambda))

        return eps_deltas

    def compute_sigma2(self, eps, delta, q, iters):
        """
        Compute sigma by doing a tenary + binary search on the moments accountant
        Faster and more accurate than compute_sigma()
        """
        def magnitude(x):
            return int(math.log10(x))

        low  = 1e-10
        high = 1e10

        while low <= high:
            mag1 = magnitude(low)
            mag2 = magnitude(high)

            if (mag2 - mag1 > 1):  # tenary search
                mid = 10**(int((mag1 + mag2)/2))
            else:
                mid = (low+high)/2

            self.accumulate_privacy_spending(sigma = mid,
                                             q     = q,
                                             iters = iters,
                                             reset = True)

            mid_eps = self.get_privacy_spent(target_deltas=[delta])[0][0]

            if eps == mid_eps:
                low = mid
                break
            elif eps > mid_eps:
                high = mid*0.99
            else:
                low = mid*1.01

        return low

    def compute_sigma(self, eps, delta, q, iters):
        """
        Compute sigma by doing a line search on the moments accountant
        """
        sigma = 0.0001
        while True:
            self.accumulate_privacy_spending(sigma = sigma,
                                             q     = q,
                                             iters = iters,
                                             reset = True)

            spent_delta = self.get_privacy_spent(target_eps=[eps])[0][1]
            if spent_delta <= delta:
                return sigma

            sigma *= 1.1

    def _privacy_accounting(self, q, iters, eps, delta=0.001):
        """
        Compute sigma if necessary and use it to compute eps
        """

        if eps > 0:
            self.sigma = self.compute_sigma2(eps   = eps,
                                             delta = delta,
                                             q     = q,
                                             iters = iters)
            print("Sigma: %g" % self.sigma)

        if self.sigma > 0:
            self.accumulate_privacy_spending(sigma = self.sigma,
                                             q     = q,
                                             iters = iters,
                                             reset = True)

            measured_eps = self.get_privacy_spent(target_deltas=[delta])[0][0]

            if abs(measured_eps - eps) > eps:
                raise(ArithmeticError("Target eps=%g whereas measured eps=%g" % (eps, measured_eps)))

            return measured_eps

        return None

    def check(self, epsilon, delta, q, iters):
        """
        Checks if the provided (epsilon,delta) can be spent without exceeding the accountant's budget ceiling by
        comparing whether the spent_budget for all the collected queries exceed the pre-defined epsilon and delta
        ceiling values or not.

        :param epsilon: float type.  Epsilon budget spend to check.
        :param delta:   float type.  Delta budget spend to check.
        :return: bool type.          True if the budget can be spent, otherwise error is raised.
        """

        check_epsilon_delta(epsilon, delta)
        if self.epsilon == float("inf") and self.delta == 1:
            return True

        if 0 < epsilon < self.__min_epsilon:
            raise ValueError(f"Epsilon must be at least {self.__min_epsilon} if non-zero, got {epsilon}.")

        measured_eps = self._privacy_accounting(q     = q,
                                                iters = iters,
                                                eps   = epsilon,
                                                delta = delta)

        measured_epsilon = measured_eps[0][0]
        spent_budget     = self.spent_budget + [(measured_epsilon, delta)]

        # Check whether the total current privacy to be spent
        if Budget(self.epsilon, self.delta, None) >= self.total(spent_budget=spent_budget):
            return measured_epsilon

        raise ValueError(f"Privacy spend of ({epsilon},{delta}) not permissible; will exceed remaining privacy budget."
                          f" Use {self.__class__.__name__}.{self.remaining.__name__}() to check remaining budget.")

    def spend(self, epsilon, delta, q, iters):
        """
        Spend the given privacy budget. Instructs the accountant to spend the given epsilon and delta privacy budget,
        while ensuring the target budget is not exceeded.

        :param epsilon: float type.  Epsilon privacy budget to be spent.
        :param delta:   float type.  Delta privacy budget to be spent.
        :return: self
        """

        # Check whether total privacy cost spent on all existing queries (considering the newly incoming query) exceed
        # the privacy budget ceiling
        measured_epsilon = self.check(epsilon = epsilon,
                                      delta   = delta,
                                      q       = q,
                                      iters   = iters)

        # If not exceeding the privacy budget ceiling, add this new query's information to __spent_budget
        self.__spent_budget.append((measured_epsilon, delta))
        return self


def GenerateBinomialTable(m):
    """
    Generate binomial table.

    Args:
        m: the size of the table.
    Returns:
        A two dimensional array T where T[i][j] = (i choose j), for 0 <= i, j <=m.
    """

    table = np.zeros((m + 1, m + 1), dtype=np.float64)

    for i in range(m + 1):
        table[i, 0] = 1

    for i in range(1, m + 1):
        for j in range(1, m + 1):
            v = table[i - 1, j] + table[i - 1, j - 1]
            assert not math.isnan(v) and not math.isinf(v)
            table[i, j] = v

    return table


class NaiveGaussianMomentsAccountant(MomentsAccountant):
    """
    MomentsAccountant which assumes Gaussian noise.

    GaussianMomentsAccountant assumes the noise added is centered Gaussian noise N(0, sigma^2 I).
    In this case, we can compute the differential moments accurately using a formula.

    For asymptotic bound, for Gaussian noise with variance sigma^2, we can show
    for L < sigma^2,  q L < sigma, log E[exp(L X)] = O(q^2 L^2 / sigma^2).

    Using this we derive that for training T epoches, with batch ratio q,
    the Gaussian mechanism with variance sigma^2 (with q < 1/sigma) is (e, d)
    private for d = exp(T/q q^2 L^2 / sigma^2 - L e). Setting L = sigma^2,
    Tq = e/2, the mechanism is (e, exp(-e sigma^2/2))-DP. Equivalently, the
    mechanism is (e, d)-DP if sigma = sqrt{2 log(1/d)}/e, q < 1/sigma,
    and T < e/(2q). This bound is better than the bound obtained using general
    composition theorems, by an Omega(sqrt{log k}) factor on epsilon, if we run
    k steps. Since we use direct estimate, the obtained privacy bound has tight
    constant.

    I1 -> E2 (Equation 4)
    I2 -> E1 (Equation 3)

    For GaussianMomentAccountant, it suffices to compute I1, as I1 >= I2,
    which reduce to computing E(P(x+s)/P(x+s-1) - 1)^i for s = 0 and 1. In the
    companion gaussian_moments.py file, we supply procedure for computing both
    I1 and I2 (the computation of I2 is through multi-precision integration
    package). It can be verified that indeed I1 >= I2 for wide range of parameters
    we have tried, though at the moment we are unable to prove this claim.

    We recommend that when using this accountant, users independently verify
    using gaussian_moments.py that for their parameters, I1 is indeed larger
    than I2. This can be done by following the instructions in
    gaussian_moments.py.
    """

    def __init__(self, moment_orders=32, verbose=True):
        """
        Initialization.

        Args:
            moment_orders: the order of moments to keep.
        """
        super(self.__class__, self).__init__(moment_orders, verbose)
        self._binomial_table = GenerateBinomialTable(self._max_moment_order)

    def _differential_moments(self, sigma, s, t):
        """
        Compute 0 to t-th differential moments for Gaussian variable.

            E[(P(x+s)/P(x+s-1)-1)^t]
          = sum_{i=0}^t (t choose i) (-1)^{t-i} E[(P(x+s)/P(x+s-1))^i]
          = sum_{i=0}^t (t choose i) (-1)^{t-i} E[exp(-i*(2*x+2*s-1)/(2*sigma^2))]
          = sum_{i=0}^t (t choose i) (-1)^{t-i} exp(i(i+1-2*s)/(2 sigma^2))
        Args:
          sigma: the noise sigma, in the multiples of the sensitivity.
          s: the shift.
          t: 0 to t-th moment.
        Returns:
          0 to t-th moment as an array of shape [t+1].
        """
        assert t <= self._max_moment_order, ("The order of %d is out "
                                             "of the upper bound %d."
                                             % (t, self._max_moment_order))

        # Compute x
        binomial = self._binomial_table[0:t+1, 0:t+1]
        signs    = np.zeros((t + 1, t + 1), dtype=np.float64)
        for i in range(t + 1):
            for j in range(t + 1):
                signs[i, j] = 1.0 - 2 * ((i - j) % 2)

        # x[i, j] = binomial[i, j] * signs[i, j] = (i choose j) * (-1)^{i-j}

        x = binomial * signs

        # Compute y
        exponents = [j * (j + 1.0 - 2.0 * s) / (2.0 * sigma * sigma) for j in range(t + 1)]

        # y[i, j] = x[i, j] * exp(exponents[j])
        #         = (i choose j) * (-1)^{i-j} * exp(j(j-1)/(2 sigma^2))
        # Note: this computation is done by broadcasting pointwise multiplication
        # between [t+1, t+1] tensor and [t+1] tensor.

        np.seterr(over='ignore', invalid='ignore')
        y = x * np.exp(exponents)

        # Compute z
        # z[i] = sum_j y[i, j]
        #      = sum_j (i choose j) * (-1)^{i-j} * exp(j(j-1)/(2 sigma^2))

        z = np.sum(y, 1)

        return z

    def _compute_log_moment(self, sigma, q, moment_order):
        """
        Compute high moment of privacy loss.

        Args:
            sigma: the noise sigma, in the multiples of the sensitivity.
            q: the sampling ratio.
            moment_order: the order of moment.
        Returns:
            log E[exp(moment_order * X)]
        """
        assert moment_order <= self._max_moment_order, ("The order of %d is out "
                                                        "of the upper bound %d."
                                                        % (moment_order, self._max_moment_order))

        # http://www.wolframalpha.com/input/?i=Solve%5Be%5E(t(t%2B1)%2F(2*sigma%5E2))+%3C+1.7976931348623157e%2B308,+sigma+%3E+0,+t+%3E+0,+sigma%5D
        #    min_sigma = 0.0265413 * np.sqrt(moment_order*(moment_order+1))
        #    assert sigma > min_sigma, ("sigma < %f => inf value for the exponential calculations" % min_sigma)

        binomial_table = self._binomial_table[moment_order:moment_order+1, 0:moment_order+1]

        # qs = [1 q q^2 ... q^L] = exp([0 1 2 ... L] * log(q))

        qs       = np.exp(np.array([range(moment_order + 1)]) * np.log(q))

        moments0 = self._differential_moments(sigma, 0.0, moment_order)
        term0    = np.sum(binomial_table * qs * moments0)

        moments1 = self._differential_moments(sigma, 1.0, moment_order)
        term1    = np.sum(binomial_table * qs * moments1)

        I1       = np.squeeze(q * term0 + (1.0 - q) * term1)

        try:
            self._E2s.append(I1) # I1 -> E2
        except AttributeError:
            self._E2s = []
            self._E2s.append(I1)

        return np.log(I1)

    def total(self, spent_budget=None):
        """
        Returns the potentially total current privacy spend for all the existing collected queries
        (queries in spent_budget + newly incoming query). spent_budget can be specified as parameters,
        otherwise the class values will be used.

        :param spent_budget: list of epsilon, optional. List of budget spends. If not provided, the accountant's
                                                                               spends will be used.

        :return: epsilon:    float type.                                            Total epsilon spend.
                 delta:      float type.                                            Total delta spend.                                       Total epsilon spend.
        """

        if spent_budget is None:
            spent_budget = self.spent_budget
        else:
            for epsilon, delta in spent_budget:
                check_epsilon_delta(epsilon, delta)

        epsilon_sum, delta_sum = 0, 0

        for epsilon, delta in spent_budget:
            epsilon_sum += epsilon
            delta_sum   += delta

        total_epsilon_naive = epsilon_sum
        total_delta_naive   = delta_sum

        return Budget(total_epsilon_naive, total_delta_naive, None)
