from numbers import Real, Integral
import numpy as np


def check_epsilon(epsilon):
    """
    Checks whether epsilon is valid value for differential privacy. An error would be thrown when failure
    happens, otherwise returns nothing.

    :param epsilon: float type;                    ${Epsilon} for differential privacy. Must be non-negative.
    :return:
    """

    if not isinstance(epsilon, Real):
        raise TypeError("Epsilon and delta must be numeric")

    if epsilon < 0:
        raise ValueError("Epsilon must be non-negative")


class Budget:
    """
    A class representing a privacy budget with only epsilon.
    """

    def __init__(self, epsilon):
        if epsilon < 0:
            raise ValueError("Epsilon must be non-negative")
        self.epsilon = epsilon

    def __gt__(self, other):
        return self.epsilon > other.epsilon

    def __ge__(self, other):
        return self.epsilon >= other.epsilon

    def __lt__(self, other):
        return self.epsilon < other.epsilon

    def __le__(self, other):
        return self.epsilon <= other.epsilon

    def __eq__(self, other):
        return self.epsilon == other.epsilon

    def __repr__(self):
        return f"Budget(epsilon={self.epsilon})"


class NaivePrivacyAccountantForPureDP:

    _default = None

    # Initialization
    # -------------------------------------------------------------------------------------------------------------
    def __init__(self,
                 epsilon      = float("inf"),
                 spent_budget = None):
        """
        Sequential composition based privacy budget accountant for ε-differential privacy.
        This class creates a privacy budget accountant to track privacy spend across queries and other data accesses.
        Once instantiated, the NaivePrivacyAccountantForPureDP stores each privacy spend and iteratively updates the
        total budget spend, raising an error when the budget ceiling (if specified) is exceeded.
        The accountant can be initialised without any maximum budget, to enable users track the total privacy spend
        of their actions without restriction.

        :param epsilon:      float type, default: infinity. Epsilon budget ceiling of the accountant.
        :param spent_budget: list epsilon, optional. List of tuples of pre-existing budget spends.  Allows for a new
                                                     accountant to be initialised with spends extracted from a previous
                                                     instance.

        """

        # Check whether the provided budget ceilings are valid
        check_epsilon(epsilon)

        # attributes starting with __ are to be protected
        self.__epsilon = epsilon
        self.__min_epsilon = 0 if epsilon == float("inf") else epsilon * 1e-14
        self.__spent_budget = []

        # If spent_budget is given, take it into account
        if spent_budget is not None:
            if not isinstance(spent_budget, list):
                raise TypeError("spent_budget must be a list")

            for _epsilon in spent_budget:
                self.spend(_epsilon)

    # Magic Methods
    # -------------------------------------------------------------------------------------------------------------
    def __repr__(self, n_budget_max=5):
        params = []
        if self.epsilon != float("inf"):
            params.append(f"epsilon={self.epsilon}")

        if self.spent_budget:
            if len(self.spent_budget) > n_budget_max:
                params.append("spent_budget=" + str(self.spent_budget[:n_budget_max] + ["..."]).replace("'", ""))
            else:
                params.append("spent_budget=" + str(self.spent_budget))

        return "BudgetAccountant(" + ", ".join(params) + ")"

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

    def set_default(self):
        """
        Sets the current accountant to be the default.
        :return: self
        """
        NaivePrivacyAccountantForPureDP._default = self
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
            if NaivePrivacyAccountantForPureDP._default is None:
                NaivePrivacyAccountantForPureDP._default = NaivePrivacyAccountantForPureDP()

            return NaivePrivacyAccountantForPureDP._default

        if not isinstance(accountant, NaivePrivacyAccountantForPureDP):
            raise TypeError(f"Accountant must be of type BudgetAccountant, got {type(accountant)}")

        return accountant

    @staticmethod
    def pop_default():
        """
        Pops the default BudgetAccountant from the class and returns it to the user.
        :return: existing default BudgetAccountant.
        """

        default = NaivePrivacyAccountantForPureDP._default
        NaivePrivacyAccountantForPureDP._default = None
        return default

    # Methods that define functions to evaluate the privacy spends
    # -------------------------------------------------------------------------------------------------------------

    def total(self, spent_budget=None):
        """
        Returns the potentially total current privacy spend for all the existing collected queries
        (queries in spent_budget + newly incoming query). spent_budget can be specified as parameters,
        otherwise the class values will be used.

        :param spent_budget: list of epsilon, optional. List of budget spends. If not provided, the accountant's
                                                                               spends will be used.
        :return: epsilon:    float type.                                       Total epsilon spend.
        """

        if spent_budget is None:
            spent_budget = self.spent_budget
        else:
            for epsilon in spent_budget:
                check_epsilon(epsilon)

        epsilon_sum = 0

        for epsilon in spent_budget:
            epsilon_sum += epsilon

        total_epsilon_naive = epsilon_sum

        return Budget(total_epsilon_naive)

    def check(self, epsilon):
        """
        Checks if the provided epsilon can be spent without exceeding the accountant's budget ceiling by comparing
        whether the spent_budget for all the collected queries exceed the pre-defined epsilon and delta ceiling values
        or not.

        :param epsilon: float type.  Epsilon budget spend to check.
        :param delta:   float type.  Delta budget spend to check.
        :return: bool type.          True if the budget can be spent, otherwise error is raised.
        """

        check_epsilon(epsilon)
        if self.epsilon == float("inf"):
            return True

        if 0 < epsilon < self.__min_epsilon:
            raise ValueError(f"Epsilon must be at least {self.__min_epsilon} if non-zero, got {epsilon}.")

        spent_costs = self.spent_budget + [epsilon]

        # Check whether the total current privacy to be spent
        if Budget(self.epsilon) >= self.total(spent_budget=spent_costs):
            return True

        raise ValueError(f"Privacy spend of ({epsilon}) not permissible; will exceed remaining privacy budget."
                         f" Use {self.__class__.__name__}.{self.remaining.__name__}() to check remaining budget.")

    def spend(self, epsilon):
        """
        Spend the given privacy budget. Instructs the accountant to spend the given epsilon privacy budget,
        while ensuring the target budget is not exceeded.

        :param epsilon: float type.  Epsilon privacy budget to be spent.
        :return: self
        """

        # Check whether total privacy cost spent on all existing queries (considering the newly incoming query) exceed
        # the privacy budget ceiling
        self.check(epsilon)

        # If not exceeding the privacy budget ceiling, add this new query's information to __spent_budget
        self.__spent_budget.append(epsilon)
        return self

    def remaining(self):
        """
        Calculates the budget that remains to be spent.

        :return: epsilon : float type.  Total epsilon remaining budget.
        """
        spent_epsilon = self.total()
        remaining_epsilon = self.epsilon - spent_epsilon.epsilon

        return Budget(remaining_epsilon)
