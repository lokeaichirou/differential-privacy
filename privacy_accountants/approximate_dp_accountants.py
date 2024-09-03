
from abc import ABC, abstractmethod
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
    def __new__(cls, epsilon, delta):
        if epsilon < 0:
            raise ValueError("Epsilon must be non-negative")

        if not 0 <= delta <= 1:
            raise ValueError("Delta must be in [0, 1]")

        return tuple.__new__(cls, (epsilon, delta))

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


class PrivacyAccountantForApproxDP:
    _default = None

    # Initialization
    # -------------------------------------------------------------------------------------------------------------
    def __init__(self,
                 epsilon=float("inf"),
                 delta=1.0,
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

        # If spent_budget is given, take it into account
        if spent_budget is not None:
            if not isinstance(spent_budget, list):
                raise TypeError("spent_budget must be a list")

            for _epsilon, _delta in spent_budget:
                self.spend(_epsilon, _delta)

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

    def check(self, epsilon, delta):
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

        spent_budget = self.spent_budget + [(epsilon, delta)]

        # Check whether the total current privacy to be spent
        if Budget(self.epsilon, self.delta) >= self.total(spent_budget=spent_budget):
            return True

        raise ValueError(f"Privacy spend of ({epsilon},{delta}) not permissible; will exceed remaining privacy budget."
                          f" Use {self.__class__.__name__}.{self.remaining.__name__}() to check remaining budget.")

    def spend(self, epsilon, delta):
        """
        Spend the given privacy budget. Instructs the accountant to spend the given epsilon and delta privacy budget,
        while ensuring the target budget is not exceeded.

        :param epsilon: float type.  Epsilon privacy budget to be spent.
        :param delta:   float type.  Delta privacy budget to be spent.
        :return: self
        """

        # Check whether total privacy cost spent on all existing queries (considering the newly incoming query) exceed
        # the privacy budget ceiling
        self.check(epsilon, delta)

        # If not exceeding the privacy budget ceiling, add this new query's information to __spent_budget
        self.__spent_budget.append((epsilon, delta))
        return self


class NaivePrivacyAccountantForApproxDP(PrivacyAccountantForApproxDP):

    def __init__(self,
                 epsilon=float("inf"),
                 delta=1.0,
                 spent_budget=None):
        """
        Sequential composition based privacy budget accountant for (ε, δ)-differential privacy.
        """

        super().__init__(epsilon      = epsilon,
                         delta        = delta,
                         spent_budget = spent_budget)

    # Methods that define functions to evaluate the privacy spends
    # -------------------------------------------------------------------------------------------------------------
    def __repr__(self, n_budget_max=5):
        params = []
        if self.epsilon != float("inf"):
            params.append(f"epsilon={self.epsilon}")

        if self.delta != 1:
            params.append(f"delta={self.delta}")

        if self.spent_budget:
            if len(self.spent_budget) > n_budget_max:
                params.append("spent_budget=" + str(self.spent_budget[:n_budget_max] + ["..."]).replace("'", ""))
            else:
                params.append("spent_budget=" + str(self.spent_budget))

        return "BudgetAccountant(" + ", ".join(params) + ")"

    def set_default(self):
        """
        Sets the current accountant to be the default.
        :return: self
        """
        NaivePrivacyAccountantForApproxDP._default = self
        return self

    # staticmethod
    # -------------------------------------------------------------------------------------------------------------
    @staticmethod
    def load_default(accountant):
        """
        Loads the default privacy budget accountant if none is supplied, otherwise checks that the supplied accountant
        is a NaivePrivacyAccountantForPureDP class. An accountant can be set as the default using the set_default()
        method. If no default has been set, a default is created.

        :param accountant: The provided budget accountant.  If None, the default accountant is returned.
        :return: working NaivePrivacyAccountantForPureDP, either the supplied `accountant` or the existing default.
        """
        if accountant is None:
            if NaivePrivacyAccountantForApproxDP._default is None:
                NaivePrivacyAccountantForApproxDP._default = NaivePrivacyAccountantForApproxDP()

            return NaivePrivacyAccountantForApproxDP._default

        if not isinstance(accountant, NaivePrivacyAccountantForApproxDP):
            raise TypeError(f"Accountant must be of type BudgetAccountant, got {type(accountant)}")

        return accountant

    @staticmethod
    def pop_default():
        """
        Pops the default BudgetAccountant from the class and returns it to the user.
        :return: existing default BudgetAccountant.
        """

        default = NaivePrivacyAccountantForApproxDP._default
        NaivePrivacyAccountantForApproxDP._default = None
        return default

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

        return Budget(total_epsilon_naive, total_delta_naive)

    def remaining(self):
        """
        Calculates the budget that remains to be spent.

        :return: epsilon : float type.  Total epsilon remaining budget.
        """
        spent_epsilon, spent_delta = self.total()
        remaining_epsilon          = self.epsilon - spent_epsilon
        remaining_delta            = self.delta - spent_delta

        return Budget(remaining_epsilon, remaining_delta)


class AdvancedPrivacyAccountantForApproxDP(PrivacyAccountantForApproxDP):

    def __init__(self,
                 epsilon=float("inf"),
                 delta=1.0,
                 slack=0.0,
                 spent_budget=None):
        """
        Advanced composition based privacy budget accountant for (ε, δ)-differential privacy.

        Implements the accountant rules as given in [KOV17]_ Kairouz, Peter, Sewoong Oh, and Pramod Viswanath.
        "The composition theorem for differential privacy." IEEE Transactions on Information Theory 63.6 (2017): 4037-4049.
        """

        self.slack          = slack
        super().__init__(epsilon=epsilon,
                         delta=delta,
                         spent_budget=spent_budget)

    # Methods that define functions to evaluate the privacy spends
    # -------------------------------------------------------------------------------------------------------------
    def __repr__(self, n_budget_max=5):
        params = []
        if self.epsilon != float("inf"):
            params.append(f"epsilon={self.epsilon}")

        if self.delta != 1:
            params.append(f"delta={self.delta}")

        if self.slack > 0:
            params.append(f"slack={self.slack}")

        if self.spent_budget:
            if len(self.spent_budget) > n_budget_max:
                params.append("spent_budget=" + str(self.spent_budget[:n_budget_max] + ["..."]).replace("'", ""))
            else:
                params.append("spent_budget=" + str(self.spent_budget))

        return "BudgetAccountant(" + ", ".join(params) + ")"

    def set_default(self):
        """
        Sets the current accountant to be the default.
        :return: self
        """
        AdvancedPrivacyAccountantForApproxDP._default = self
        return self

    # staticmethod
    # -------------------------------------------------------------------------------------------------------------
    @staticmethod
    def load_default(accountant):
        """
        Loads the default privacy budget accountant if none is supplied, otherwise checks that the supplied accountant
        is a NaivePrivacyAccountantForPureDP class. An accountant can be set as the default using the set_default()
        method. If no default has been set, a default is created.

        :param accountant: The provided budget accountant.  If None, the default accountant is returned.
        :return: working NaivePrivacyAccountantForPureDP, either the supplied `accountant` or the existing default.
        """
        if accountant is None:
            if AdvancedPrivacyAccountantForApproxDP._default is None:
                AdvancedPrivacyAccountantForApproxDP._default = AdvancedPrivacyAccountantForApproxDP()

            return AdvancedPrivacyAccountantForApproxDP._default

        if not isinstance(accountant, AdvancedPrivacyAccountantForApproxDP):
            raise TypeError(f"Accountant must be of type BudgetAccountant, got {type(accountant)}")

        return accountant

    @staticmethod
    def pop_default():
        """
        Pops the default BudgetAccountant from the class and returns it to the user.
        :return: existing default BudgetAccountant.
        """

        default = AdvancedPrivacyAccountantForApproxDP._default
        AdvancedPrivacyAccountantForApproxDP._default = None
        return default

    @staticmethod
    def __total_delta_safe(spent_budget, slack):
        """
        Calculate total delta spend of spent_budget, with special consideration for floating point arithmetic.

        :param spent_budget: list of tuples of the form (epsilon, delta). List of budget spends, for which the total
                                                                          delta spend is to be calculated.

        :param slack: float type.                                         Delta slack parameter for composition of
                                                                          spends.
        :return: Total delta spend.
        """

        # By following the Theorem 3.5 in Section 3.3 Composition Theorem for Heterogeneous Mechanisms in [KOV17],
        # Suppose there are k queries in total, then composed_delta = 1 - (1 - slack)* \prod_{l=1}^{l=k} (1 - delta_l)

        delta_spend = [slack]
        for _, delta in spent_budget:
            delta_spend.append(delta)
        delta_spend.sort()

        composed_delta = 1
        for delta in delta_spend:
            composed_delta *= 1 - delta
        composed_delta = 1 - composed_delta

        return composed_delta

    def total(self, spent_budget=None, slack=None):
        """
        Returns the potentially total current privacy spend for all the existing collected queries
        (queries in spent_budget + newly incoming query). spent_budget and slack can be specified as parameters,
        otherwise the class values will be used.

        :param spent_budget: list of tuples of the form (epsilon, delta), optional. List of tuples of budget spends.
                                                                                    If not provided, the accountant's
                                                                                    spends will be used.
        :param slack:        float type, optional.                                  Slack in delta for composition. If
                                                                                    not provided, the accountant's slack
                                                                                    will be used.
        :return: epsilon:    float type.                                            Total epsilon spend.
                 delta:      float type.                                            Total delta spend.
        """

        if spent_budget is None:
            spent_budget = self.spent_budget
        else:
            for epsilon, delta in spent_budget:
                check_epsilon_delta(epsilon, delta)

        if slack is None:
            slack = self.slack
        elif not 0 <= slack <= self.delta:
            raise ValueError(f"Slack must be between 0 and delta ({self.delta}), inclusive. Got {slack}.")

        epsilon_sum, epsilon_sq_sum, epsilon_exp_sum, sub_term_in_second_term, sub_term_in_third_term = 0, 0, 0, 0, 0

        # By following the Theorem 3.5 in Section 3.3 Composition Theorem for Heterogeneous Mechanisms in [KOV17],
        # the composed epsilon is the minimum values in the set of three terms

        for epsilon, _ in spent_budget:
            epsilon_sum += epsilon
            epsilon_sq_sum += epsilon ** 2
            epsilon_exp_sum += (np.exp(epsilon) - 1) * epsilon / (np.exp(epsilon) + 1)

        total_epsilon_naive = epsilon_sum
        total_delta = self.__total_delta_safe(spent_budget, slack)

        # If slack = 0, we don't consider the other two terms as slack is the denominator in these two terms
        if slack == 0:
            return Budget(total_epsilon_naive, total_delta)

        for epsilon, _ in spent_budget:
            sub_term_in_second_term += 2 * epsilon ** 2 * np.log(np.exp(1) + np.sqrt(epsilon_sq_sum) / slack)
            sub_term_in_third_term += 2 * epsilon ** 2 * np.log(1 / slack)

        second_term = epsilon_exp_sum + np.sqrt(sub_term_in_second_term)
        third_term = epsilon_exp_sum + np.sqrt(sub_term_in_third_term)

        return Budget(min(total_epsilon_naive, second_term, third_term), total_delta)

    def remaining(self, k=1):
        """
        Calculates the budget that remains to be spent. Calculates the privacy budget that can be spent on k queries.
        Spending this budget on k queries will match the budget ceiling, assuming no floating point errors.

        :param k: int type, default: 1. The number of queries for which to calculate the remaining budget.

        :return: epsilon : float type.  Total epsilon spend remaining for k queries.
                 delta :   float type.  Total delta spend remaining for k queries.
        """

        if not isinstance(k, Integral):
            raise TypeError(f"k must be integer-valued, got {type(k)}.")
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}.")

        _, spent_delta = self.total()
        delta = 1 - ((1 - self.delta) / (1 - spent_delta)) ** (1 / k) if spent_delta < 1.0 else 1.0

        lower = 0
        upper = self.epsilon
        old_interval_size = (upper - lower) * 2

        while old_interval_size > upper - lower:
            old_interval_size = upper - lower
            mid = (upper + lower) / 2

            spent_budget = self.spent_budget + [(mid, 0)] * k
            x_0, _ = self.total(spent_budget=spent_budget)

            if x_0 >= self.epsilon:
                upper = mid
            if x_0 <= self.epsilon:
                lower = mid

        epsilon = (upper + lower) / 2

        return Budget(epsilon, delta)
