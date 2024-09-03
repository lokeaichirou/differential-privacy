"""
Base classes for differential privacy mechanisms.

This module provides base classes for differential privacy mechanisms. It defines the abstract base class `DPMechanism`
and the mixin class `TruncationAndFoldingMixin`. These classes serve as the foundation for implementing specific
differential privacy mechanisms.

Classes:
- DPMechanism: Abstract base class for all differential privacy mechanisms.
- TruncationAndFoldingMixin: Mixin class for truncating or folding the outputs of a mechanism.
"""

import abc
import inspect
from numbers import Real
from utils import check_random_state


class DPMechanism(abc.ABC):
    """
    Abstract base class for all differential privacy mechanisms.

    This class defines the common interface and behavior for differential privacy mechanisms. It provides methods for
    initializing the mechanism, checking the validity of privacy parameters, and randomizing values.

    Attributes:
    - epsilon: Privacy parameter epsilon for the mechanism. Must be in [0, ∞].
    - delta: Privacy parameter delta for the mechanism. Must be in [0, 1]. Cannot be simultaneously zero with epsilon.
    - random_state: Controls the randomness of the mechanism. To obtain a deterministic behavior during randomization,
      random_state has to be fixed to an integer.

    Methods:
    - randomise(value): Abstract method that randomizes the given value with the mechanism.

    Class Methods:
    - _check_epsilon_delta(epsilon, delta): Checks the validity of epsilon and delta privacy parameters.

    Private Methods:
    - _check_all(value): Checks that all parameters of the mechanism have been initialized correctly.

    Special Methods:
    - __repr__(): Returns a string representation of the mechanism object.
    """

    def __init__(self, epsilon, delta, random_state=None):
        """
        Initializes a DPMechanism object.

        Parameters:
        - epsilon: Privacy parameter epsilon for the mechanism. Must be in [0, ∞].
        - delta: Privacy parameter delta for the mechanism. Must be in [0, 1]. Cannot be simultaneously zero with epsilon.
        - random_state: Controls the randomness of the mechanism. To obtain a deterministic behavior during randomization,
          random_state has to be fixed to an integer.
        """

        self.epsilon, self.delta = self._check_epsilon_delta(epsilon, delta)
        self.random_state = random_state
        self._rng = check_random_state(random_state, True)

    def __repr__(self):
        """
        Returns a string representation of the mechanism object.

        Returns:
        - String representation of the mechanism object.
        """

        attrs = inspect.getfullargspec(self.__class__).kwonlyargs
        attr_output = []

        for attr in attrs:
            attr_output.append(attr + "=" + repr(self.__getattribute__(attr)))

        return str(self.__module__) + "." + str(self.__class__.__name__) + "(" + ", ".join(attr_output) + ")"

    @abc.abstractmethod
    def randomise(self, value):
        """
        Abstract method that randomizes the given value with the mechanism.

        Parameters:
        - value: The value to be randomized.

        Returns:
        - The randomized value with the same type as the input value.
        """

    @classmethod
    def _check_epsilon_delta(cls, epsilon, delta):
        """
        Checks the validity of epsilon and delta privacy parameters.

        Parameters:
        - epsilon: Privacy parameter epsilon.
        - delta: Privacy parameter delta.

        Returns:
        - Tuple containing the validated epsilon and delta values.

        Raises:
        - TypeError: If epsilon or delta is not a numeric type.
        - ValueError: If epsilon is not positive or delta is not in [0, 1], or if epsilon and delta are both zero.
        """

        if not isinstance(epsilon, Real) or not isinstance(delta, Real):
            raise TypeError("Epsilon and delta must be numeric")

        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")

        if not 0 <= delta <= 1:
            raise ValueError("Delta must be in [0, 1]")

        if epsilon + delta == 0:
            raise ValueError("Epsilon and Delta cannot both be zero")

        return float(epsilon), float(delta)

    def _check_all(self, value):
        """
        Checks that all parameters of the mechanism have been initialized correctly.

        Parameters:
        - value: The value to be checked.

        Returns:
        - True if all parameters are valid, False otherwise.
        """

        del value
        self._check_epsilon_delta(self.epsilon, self.delta)

        return True


class TruncationAndFoldingMixin:
    """
    Mixin class for truncating or folding the outputs of a mechanism.

    This class provides methods for truncating or folding the outputs of a mechanism. It is intended to be used as a
    mixin together with the DPMechanism class.

    Attributes:
    - lower: The lower bound of the mechanism.
    - upper: The upper bound of the mechanism.

    Methods:
    - _truncate(value): Truncates the value to be within the lower and upper bounds.
    - _fold(value): Folds the value to be within the lower and upper bounds.

    Private Methods:
    - _check_bounds(lower, upper): Performs a check on the bounds provided for the mechanism.
    - _check_all(value): Checks that all parameters of the mechanism have been initialized correctly.
    """

    def __init__(self, lower, upper):
        """
        Initializes a TruncationAndFoldingMixin object.

        Parameters:
        - lower: The lower bound of the mechanism.
        - upper: The upper bound of the mechanism.
        """

        if not isinstance(self, DPMechanism):
            raise TypeError("TruncationAndFoldingMachine must be implemented together with DPMechanism`")

        self.lower, self.upper = self._check_bounds(lower, upper)

    @classmethod
    def _check_bounds(cls, lower, upper):
        """
        Performs a check on the bounds provided for the mechanism.

        Parameters:
        - lower: The lower bound of the mechanism.
        - upper: The upper bound of the mechanism.

        Returns:
        - Tuple containing the validated lower and upper bounds.

        Raises:
        - TypeError: If lower or upper is not a numeric type.
        - ValueError: If lower is greater than upper.
        """

        if not isinstance(lower, Real) or not isinstance(upper, Real):
            raise TypeError("Bounds must be numeric")

        if lower > upper:
            raise ValueError("Lower bound must not be greater than upper bound")

        return lower, upper

    def _check_all(self, value):
        """
        Checks that all parameters of the mechanism have been initialized correctly.

        Parameters:
        - value: The value to be checked.

        Returns:
        - True if all parameters are valid, False otherwise.
        """

        del value
        self._check_bounds(self.lower, self.upper)

        return True

    def _truncate(self, value):
        """
        Truncates the value to be within the lower and upper bounds.

        Parameters:
        - value: The value to be truncated.

        Returns:
        - The truncated value.
        """

        if value > self.upper:
            return self.upper

        if value < self.lower:
            return self.lower

        return value

    def _fold(self, value):
        """
        Folds the value to be within the lower and upper bounds.

        Parameters:
        - value: The value to be folded.

        Returns:
        - The folded value.
        """

        if value < self.lower:
            return self._fold(2 * self.lower - value)

        if value > self.upper:
            return self._fold(2 * self.upper - value)

        return value
