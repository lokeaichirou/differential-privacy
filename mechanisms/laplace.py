"""
The classic Laplace mechanism in differential privacy, and its derivatives.
"""

from numbers import Real
import numpy as np
from mechanisms.base import DPMechanism, TruncationAndFoldingMixin
from utils import copy_docstring


class Laplace(DPMechanism):
    def __init__(self, *,
                 epsilon,
                 delta        = 0.0,
                 sensitivity,
                 random_state = None):
        """
        The classical Laplace mechanism in differential privacy.

        :param epsilon:      float type; Privacy parameter ${epsilon} for the mechanism. Must be in [0, ∞].
        :param delta:        float type, default: 0.0; Privacy parameter ${delta} for the mechanism. Must be in [0, 1].
                             Cannot be simultaneously zero with ${epsilon}.
        :param sensitivity:  float type; The sensitivity of the mechanism. Must be in [0, ∞).
        :param random_state: int type or RandomState, optional; Controls the randomness of the mechanism.  To obtain a
                             deterministic behaviour during randomisation, random_state has to be fixed to an integer.

        """

        super().__init__(epsilon      = epsilon,
                         delta        = delta,
                         random_state = random_state)

        self.sensitivity = self._check_sensitivity(sensitivity)
        self._scale      = None

    @classmethod
    def _check_sensitivity(cls, sensitivity):
        """
        Check if the sensitivity is valid.

        :param sensitivity: The sensitivity value to be checked.
        :return: The validated sensitivity value.
        :raises TypeError: If the sensitivity is not a numeric value.
        :raises ValueError: If the sensitivity is negative.
        """
        if not isinstance(sensitivity, Real):
            raise TypeError("Sensitivity must be numeric")

        if sensitivity < 0:
            raise ValueError("Sensitivity must be non-negative")

        return float(sensitivity)

    def _check_all(self, value):
        """
        Check if the value and sensitivity are valid.

        :param value: The value to be checked.
        :return: True if the value and sensitivity are valid.
        :raises TypeError: If the value is not a number.
        """
        super()._check_all(value)
        self._check_sensitivity(self.sensitivity)

        if not isinstance(value, Real):
            raise TypeError("Value to be randomised must be a number")

        return True

    def randomise(self, value):
        """
        Randomise given value with certain mechanism.

        :param value: The value to be randomised.
        :return: The randomised value.
        """
        self._check_all(value)

        scale            = self.sensitivity / self.epsilon
        laplace_noise    = np.random.laplace(loc=0, scale=scale)

        return value + laplace_noise


class LaplaceTruncated(Laplace, TruncationAndFoldingMixin):

    def __init__(self,
                 *,
                 epsilon,
                 delta        = 0.0,
                 sensitivity,
                 lower,
                 upper,
                 random_state = None):

        """
        The truncated Laplace mechanism, where values outside the domain are mapped to the closest point within the domain before adding noise to the value.

        :param epsilon:      float type.               Privacy parameter ${epsilon} for the mechanism. Must be in [0, ∞].
        :param delta:        float type, default: 0.0. Privacy parameter ${delta} for the mechanism. Must be in [0, 1].
                                                       Cannot be simultaneously zero with ${epsilon}.
        :param sensitivity:  float type.               The sensitivity of the mechanism. Must be in [0, ∞).

        :param lower:        float type.               The lower bound of the mechanism.

        :param upper:        float type.               The upper bound of the mechanism.
        :param random_state: int or RandomState, optional. Controls the randomness of the mechanism. To obtain a
                                                           deterministic behaviour during randomisation, random_state
                                                           has to be fixed to an integer.
        """
        super().__init__(epsilon       = epsilon,
                         delta         = delta,
                         sensitivity   = sensitivity,
                         random_state  = random_state)

        TruncationAndFoldingMixin.__init__(self, lower=lower, upper=upper)

    def _check_all(self, value):
        """
        Check if the value, sensitivity, and truncation bounds are valid.

        :param value: The value to be checked.
        :return: True if the value, sensitivity, and truncation bounds are valid.
        :raises TypeError: If the value is not a number.
        """
        Laplace._check_all(self, value)
        TruncationAndFoldingMixin._check_all(self, value)

        return True

    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        """
        Randomise given value with certain mechanism.

        :param value: The value to be randomised.
        :return: The randomised value.
        """
        self._check_all(value)
        noisy_value = super().randomise(value)

        return self._truncate(noisy_value)


class LaplaceFolded(Laplace, TruncationAndFoldingMixin):

    def __init__(self,
                 *,
                 epsilon,
                 delta        = 0.0,
                 sensitivity,
                 lower,
                 upper,
                 random_state = None):

        """
        The folded Laplace mechanism, values outside the range are to be folded to be within the range.

        :param epsilon:      float type; Privacy parameter ${epsilon} for the mechanism. Must be in [0, ∞].
        :param delta:        float type, default: 0.0; Privacy parameter ${delta} for the mechanism. Must be in [0, 1].
                                        Cannot be simultaneously zero with ${epsilon}.
        :param sensitivity:  float type; The sensitivity of the mechanism. Must be in [0, ∞).
        :param lower:        float type. The lower bound of the mechanism.
        :param upper:        float type. The upper bound of the mechanism.
        :param random_state: int or RandomState, optional. Controls the randomness of the mechanism. To obtain a
                                                           deterministic behaviour during randomisation, random_state
                                                           has to be fixed to an integer.
        """

        super().__init__(epsilon       = epsilon,
                         delta         = delta,
                         sensitivity   = sensitivity,
                         random_state  = random_state)

        TruncationAndFoldingMixin.__init__(self, lower=lower, upper=upper)

    def _check_all(self, value):
        """
        Check if the value, sensitivity, and truncation bounds are valid.

        :param value: The value to be checked.
        :return: True if the value, sensitivity, and truncation bounds are valid.
        :raises TypeError: If the value is not a number.
        """
        super()._check_all(value)
        TruncationAndFoldingMixin._check_all(self, value)

        return True

    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        """
        Randomise given value with certain mechanism.

        :param value: The value to be randomised.
        :return: The randomised value.
        """
        self._check_all(value)
        noisy_value = super().randomise(value)

        return self._fold(noisy_value)
