"""
Implementation of the standard exponential mechanism
"""

from numbers import Real
import numpy as np
from mechanisms.base import DPMechanism


class Exponential(DPMechanism):

    def __init__(self,
                 *,
                 epsilon,
                 sensitivity,
                 utility,
                 candidates   = None,
                 measure      = None,
                 random_state = None):

        """
        The exponential mechanism for achieving differential privacy on candidate selection, as first proposed by McSherry
        and Talwar. The exponential mechanism achieves differential privacy by randomly choosing a candidate subject to
        candidate utility scores, with greater probability given to higher-utility candidates.

        :param epsilon:      float type; Privacy parameter ${epsilon} for the mechanism. Must be in (0, ∞].
        :param sensitivity:  float type; The sensitivity in utility values to a change in a datapoint in the underlying dataset.
        :param utility:      list type;  A list of non-negative utility values for each candidate.
        :param monotonic:    bool type; default: False. Specifies if the utility function is monotonic, i.e. that adding
                                                        an individual to the underlying dataset can only increase the
                                                        values in `utility`.
        :param candidates:   list type; optional. An optional list of candidate labels. If omitted, the zero-indexed
                                                  list [0, 1, ..., n] is used.
        :param measure:      list type; optional. An optional list of measures for each candidate. If omitted, a uniform
                                                  measure is used.
        :param random_state: int type or RandomState, optional; Controls the randomness of the mechanism.  To obtain a
                             deterministic behaviour during randomisation, random_state has to be fixed to an integer.

        References
        ----------
           [MT07] Frank McSherry and Kunal Talwar. Mechanism design via differential privacy. In Proceedings of the 48th
           Annual IEEE Symposium on Foundations of Computer Science, FOCS ’07, pages 94–103, Washington, DC, USA, 2007.
           IEEE Computer Society.

        """

        super().__init__(epsilon      = epsilon,
                         delta        = 0.0,
                         random_state = random_state)

        self.sensitivity                            = self._check_sensitivity(sensitivity)
        self.utility, self.candidates, self.measure = self._check_utility_candidates_measure(utility,
                                                                                             candidates,
                                                                                             measure)
        self._probabilities                         = self._find_probabilities(self.epsilon,
                                                                               self.sensitivity,
                                                                               self.utility,
                                                                               self.measure)

    @classmethod
    def _check_epsilon_delta(cls, epsilon, delta):
        if not delta == 0:
            raise ValueError("Delta must be zero")

        return super()._check_epsilon_delta(epsilon, delta)

    @classmethod
    def _check_sensitivity(cls, sensitivity):
        if not isinstance(sensitivity, Real):
            raise TypeError("Sensitivity must be numeric")

        if sensitivity < 0:
            raise ValueError("Sensitivity must be non-negative")

        return float(sensitivity)

    @classmethod
    def _check_utility_candidates_measure(cls, utility, candidates, measure):
        if not isinstance(utility, list):
            raise TypeError(f"Utility must be a list, got a {utility}.")

        if not all(isinstance(u, Real) for u in utility):
            raise TypeError("Utility must be a list of real-valued numbers.")

        if len(utility) < 1:
            raise ValueError("Utility must have at least one element.")

        if np.isinf(utility).any():
            raise ValueError("Utility must be a list of finite numbers.")

        if candidates is not None:
            if not isinstance(candidates, list):
                raise TypeError(f"Candidates must be a list, got a {type(candidates)}.")

            if len(candidates) != len(utility):
                raise ValueError("List of candidates must be the same length as the list of utility values.")

        if measure is not None:
            if not isinstance(measure, list):
                raise TypeError(f"Measure must be a list, got a {type(measure)}.")

            if not all(isinstance(m, Real) for m in measure):
                raise TypeError("Measure must be a list of real-valued numbers.")

            if np.isinf(measure).any():
                raise ValueError("Measure must be a list of finite numbers.")

            if len(measure) != len(utility):
                raise ValueError("List of measures must be the same length as the list of utility values.")

        return utility, candidates, measure

    @classmethod
    def _find_probabilities(cls,
                            epsilon,
                            sensitivity,
                            utility,
                            measure):

        scale = epsilon / sensitivity / 2 if sensitivity / epsilon > 0 else float("inf")

        # Set max utility to 0 to avoid overflow on high utility
        utility        = np.array(utility) - max(utility)
        probabilities  = np.isclose(utility, 0).astype(float) if np.isinf(scale) else np.exp(scale * utility)
        probabilities *= np.array(measure) if measure else 1

        # Normalization
        probabilities /= probabilities.sum()

        return np.cumsum(probabilities)

    def _check_all(self, value):
        super()._check_all(value)
        self._check_sensitivity(self.sensitivity)
        self._check_utility_candidates_measure(self.utility, self.candidates, self.measure)

        if value is not None:
            raise ValueError(f"Value to be randomised must be None. Got: {value}.")

        return True

    def randomise(self, value=None):
        """
        Select a candidate with differential privacy.

        :param value:
        :return: The randomised candidate.
        """

        self._check_all(value)
        rand = self._rng.random()

        if np.any(rand <= self._probabilities):
            idx = np.argmax(rand <= self._probabilities)
        elif np.isclose(rand, self._probabilities[-1]):
            idx = len(self._probabilities) - 1
        else:
            raise RuntimeError("Can't find a candidate to return. "
                               f"Debugging info: Rand: {rand}, Probabilities: {self._probabilities}")

        return self.candidates[idx] if self.candidates else idx
