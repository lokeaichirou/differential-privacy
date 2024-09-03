
from abc import ABCMeta
from abc import abstractmethod
import warnings
import numbers
import numpy as np
import math
from time import time

from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import issparse

from sklearn.model_selection import train_test_split

from sklearn.tree._tree import DTYPE

from sklearn.base import is_classifier
from sklearn.base import BaseEstimator

from sklearn.utils import check_array, check_random_state, column_or_1d
from sklearn.utils.validation import check_is_fitted, _check_sample_weight

from sklearn.ensemble._base import BaseEnsemble
from sklearn.ensemble import _gb_losses
from sklearn.ensemble._gradient_boosting import _random_sample_mask, predict_stage, predict_stages

from models.gradient_boosting_decision_trees.dp_tree import DifferentiallyPrivateDecisionTreeRegressor
from tailed_notions import *


class VerboseReporter:
    """
    Reports verbose output to stdout.

    Parameters
    ----------
    verbose : int
        Verbosity level. If ``verbose==1`` output is printed once in a while
        (when iteration mod verbose_mod is zero).; if larger than 1 then output
        is printed for each update.
    """
    def __init__(self, verbose):
        self.verbose = verbose

    def init(self, est, begin_at_stage=0):
        """Initialize reporter

        Parameters
        ----------
        est : Estimator
            The estimator

        begin_at_stage : int, default=0
            stage at which to begin reporting
        """
        # header fields and line format str
        header_fields = ['Iter', 'Train Loss']
        verbose_fmt   = ['{iter:>10d}', '{train_score:>16.4f}']
        # do oob?
        if est.subsample < 1:
            header_fields.append('OOB Improve')
            verbose_fmt.append('{oob_impr:>16.4f}')

        header_fields.append('Remaining Time')
        verbose_fmt.append('{remaining_time:>16s}')

        # print the header line
        print(('%10s ' + '%16s ' * (len(header_fields) - 1)) % tuple(header_fields))

        self.verbose_fmt    = ' '.join(verbose_fmt)

        # plot verbose info each time i % verbose_mod == 0
        self.verbose_mod    = 1
        self.start_time     = time()
        self.begin_at_stage = begin_at_stage

    def update(self, j, est):
        """
        Update reporter with new iteration.

        Parameters
        ----------
        j : int
            The new iteration.
        est : Estimator
            The estimator.
        """
        do_oob = est.subsample < 1
        # we need to take into account if we fit additional estimators.
        i      = j - self.begin_at_stage  # iteration relative to the start iter
        if (i + 1) % self.verbose_mod == 0:
            oob_impr = est.oob_improvement_[j] if do_oob else 0
            remaining_time = ((est.n_estimators - (j + 1)) * (time() - self.start_time) / float(i + 1))
            if remaining_time > 60:
                remaining_time = '{0:.2f}m'.format(remaining_time / 60.0)
            else:
                remaining_time = '{0:.2f}s'.format(remaining_time)
            print(self.verbose_fmt.format(iter           = j + 1,
                                          train_score    = est.train_score_[j],
                                          oob_impr       = oob_impr,
                                          remaining_time =remaining_time))

            if self.verbose == 1 and ((i + 1) // (self.verbose_mod * 10) > 0):
                # adjust verbose frequency (powers of 10)
                self.verbose_mod *= 10


class BaseDifferentiallyPrivateGradientBoosting(BaseEnsemble, metaclass=ABCMeta):
    """Abstract base class for differentially private Gradient Boosting."""

    @abstractmethod
    def __init__(self,
                 *,
                 privacy_notion,
                 loss,
                 learning_rate,
                 n_estimators,
                 min_samples_split,
                 max_depth,
                 init,
                 subsample,
                 random_state,
                 verbose,
                 cat_idx,
                 l2_threshold = 1.0,
                 l2_lambda    = 0.1,
                 warm_start=False,
                 validation_fraction=0.1,
                 n_iter_no_change=None,
                 tol=1e-4):

        self.privacy_notion      = privacy_notion
        self.loss                = loss
        self.learning_rate       = learning_rate
        self.n_estimators        = n_estimators
        self.min_samples_split   = min_samples_split
        self.max_depth           = max_depth
        self.init                = init
        self.subsample           = subsample
        self.random_state        = random_state
        self.verbose             = verbose
        self.cat_idx             = cat_idx
        self.warm_start          = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change    = n_iter_no_change
        self.tol                 = tol
        self.l2_threshold        = l2_threshold
        self.l2_lambda           = l2_lambda

        self.parse_privacy_notion()

    def parse_privacy_notion(self):
        # Parse privacy_notion
        if isinstance(self.privacy_notion, EpsilonDp):
            self.epsilon = self.privacy_notion.epsilon
            self.delta   = 0

        elif isinstance(self.privacy_notion, EpsilonDeltaDp):
            self.epsilon = self.privacy_notion.epsilon
            self.delta   = self.privacy_notion.delta

        elif isinstance(self.privacy_notion, RenyiDp):
            self.alpha       = self.privacy_notion.alpha
            self.epsilon_bar = self.privacy_notion.epsilon_bar
            self.delta       = self.privacy_notion.delta
            self.epsilon     = self.epsilon_bar + np.log(1 / self.delta) / (self.alpha - 1)

    @abstractmethod
    def _validate_y(self, y, sample_weight=None):
        """Called by fit to validate y."""

    def _check_params(self):
        """Check validity of parameters and raise ValueError if not valid."""
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0 but "
                             "was %r" % self.n_estimators)

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than 0 but "
                             "was %r" % self.learning_rate)

        if (self.loss not in self._SUPPORTED_LOSS
                or self.loss not in _gb_losses.LOSS_FUNCTIONS):
            raise ValueError("Loss '{0:s}' not supported. ".format(self.loss))

        if self.loss == 'deviance':
            loss_class = (_gb_losses.MultinomialDeviance
                          if len(self.classes_) > 2
                          else _gb_losses.BinomialDeviance)
        else:
            loss_class = _gb_losses.LOSS_FUNCTIONS[self.loss]

        if is_classifier(self):
            self.loss_ = loss_class(self.n_classes_)
        else:
            self.loss_ = loss_class()

        if not (0.0 < self.subsample <= 1.0):
            raise ValueError("subsample must be in (0,1] but "
                             "was %r" % self.subsample)

        if self.init is not None:
            # init must be an estimator or 'zero'
            if isinstance(self.init, BaseEstimator):
                self.loss_.check_init_estimator(self.init)
            elif not (isinstance(self.init, str) and self.init == 'zero'):
                raise ValueError("The init parameter must be an estimator or 'zero'. "
                                 "Got init={}".format(self.init))

        if not isinstance(self.n_iter_no_change, (numbers.Integral, type(None))):
            raise ValueError("n_iter_no_change should either be None or an "
                             "integer. %r was passed" % self.n_iter_no_change)

    def _init_state(self):
        """Initialize model state and allocate model state data structures. """

        self.init_ = self.init
        if self.init_ is None:
            self.init_ = self.loss_.init_estimator()

        self.estimators_  = np.empty((self.n_estimators, self.loss_.K), dtype=object)
        self.train_score_ = np.zeros((self.n_estimators,), dtype=np.float64)
        # do oob?
        if self.subsample < 1.0:
            self.oob_improvement_ = np.zeros((self.n_estimators), dtype=np.float64)

    def _clear_state(self):
        """Clear the state of the gradient boosting model. """
        if hasattr(self, 'estimators_'):
            self.estimators_ = np.empty((0, 0), dtype=object)
        if hasattr(self, 'train_score_'):
            del self.train_score_
        if hasattr(self, 'oob_improvement_'):
            del self.oob_improvement_
        if hasattr(self, 'init_'):
            del self.init_
        if hasattr(self, '_rng'):
            del self._rng

    def _resize_state(self):
        """Add additional ``n_estimators`` entries to all attributes."""
        # self.n_estimators is the number of additional est to fit
        total_n_estimators = self.n_estimators
        if total_n_estimators < self.estimators_.shape[0]:
            raise ValueError('resize with smaller n_estimators %d < %d' % (total_n_estimators, self.estimators_[0]))

        self.estimators_  = np.resize(self.estimators_, (total_n_estimators, self.loss_.K))
        self.train_score_ = np.resize(self.train_score_, total_n_estimators)

        if self.subsample < 1 or hasattr(self, 'oob_improvement_'):
            # if do oob resize arrays or create new if not available
            if hasattr(self, 'oob_improvement_'):
                self.oob_improvement_ = np.resize(self.oob_improvement_, total_n_estimators)
            else:
                self.oob_improvement_ = np.zeros((total_n_estimators,), dtype=np.float64)

    def _is_initialized(self):
        return len(getattr(self, 'estimators_', [])) > 0

    def _check_initialized(self):
        """Check that the estimator is initialized, raising an error if not."""
        check_is_fitted(self)

    def _raw_predict_init(self, X):
        """Check input and compute raw predictions of the init estimator."""
        self._check_initialized()
        X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)

        if X.shape[1] != self.n_features_:
            raise ValueError("X.shape[1] should be {0:d}, not {1:d}.".format(self.n_features_, X.shape[1]))
        if self.init_ == 'zero':
            raw_predictions = np.zeros(shape=(X.shape[0], self.loss_.K), dtype=np.float64)
        else:
            raw_predictions = self.loss_.get_init_raw_predictions(X, self.init_).astype(np.float64)

        return raw_predictions

    def _raw_predict(self, X):
        """Return the sum of the trees raw predictions (+ init estimator)."""
        raw_predictions = self._raw_predict_init(X)
        predict_stages(self.estimators_, X,
                       self.learning_rate,
                       raw_predictions)
        return raw_predictions

    def _staged_raw_predict(self, X):
        """
        Compute raw predictions of ``X`` for each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        raw_predictions : generator of ndarray of shape (n_samples, k)
            The raw predictions of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
            Regression and binary classification are special cases with
            ``k == 1``, otherwise ``k==n_classes``.
        """
        X               = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
        raw_predictions = self._raw_predict_init(X)
        for i in range(self.estimators_.shape[0]):
            predict_stage(self.estimators_,
                          i,
                          X,
                          self.learning_rate,
                          raw_predictions)
            yield raw_predictions.copy()

    def _fit_stage(self,
                   i,
                   X,
                   y,
                   raw_predictions,
                   sample_weight,
                   sample_mask,
                   random_state,
                   X_csc=None,
                   X_csr=None):

        """
        Fit another stage of ``_n_classes`` trees to the boosting model.

        """

        assert sample_mask.dtype == bool
        loss                      = self.loss_
        original_y                = y

        for k in range(loss.K):

            if loss.is_multi_class:
                y = np.array(original_y == k, dtype=np.float64)

            tree_index     = i * loss.K + k
            number_of_rows = int((len(X) * self.learning_rate * math.pow((1 - self.learning_rate), tree_index)) / (1 - math.pow((1 - self.learning_rate), loss.K)))
            rows           = np.random.randint(len(X), size=number_of_rows)

            X_tree               = X[rows, :]
            y_tree               = y[rows]
            raw_predictions_tree = raw_predictions[rows]

            residual = loss.negative_gradient(y,
                                              raw_predictions,
                                              k = k,
                                              sample_weight = sample_weight)

            residual_tree = loss.negative_gradient(y_tree,
                                                   raw_predictions_tree,
                                                   k=k,
                                                   sample_weight=sample_weight)

            # induce regression tree on residuals
            tree = DifferentiallyPrivateDecisionTreeRegressor(tree_index        = tree_index,
                                                              epsilon           = self.epsilon/self.n_estimators,
                                                              max_depth         = self.max_depth,
                                                              max_leaves        = None,
                                                              min_samples_split = self.min_samples_split,
                                                              leaf_clipping     = True,
                                                              learning_rate     = self.learning_rate,
                                                              l2_threshold      = self.l2_threshold,
                                                              l2_lambda         = self.l2_lambda,
                                                              delta_g           = 3 * np.square(self.l2_threshold),
                                                              delta_v           = min(self.l2_threshold / (1 + self.l2_lambda),
                                                                                      2 * self.l2_threshold * math.pow((1 - self.learning_rate), tree_index)),
                                                              cat_idx           = self.cat_idx)

            if self.subsample < 1.0:
                # no inplace multiplication!
                sample_weight = sample_weight * sample_mask.astype(np.float64)

            X = X_csr if X_csr is not None else X
            tree.fit(X, residual)

            # update tree leaves
            loss.update_terminal_regions(tree.tree_,
                                         X,
                                         y,
                                         residual,
                                         raw_predictions,
                                         sample_weight,
                                         sample_mask,
                                         learning_rate=self.learning_rate,
                                         k=k)

            # add tree to ensemble
            self.estimators_[i, k] = tree

        return raw_predictions

    def _fit_stages(self,
                    X,
                    y,
                    raw_predictions,
                    sample_weight,
                    random_state,
                    X_val,
                    y_val,
                    sample_weight_val,
                    begin_at_stage=0,
                    monitor=None):

        """
        Iteratively fits the stages.

        For each stage it computes the progress (OOB, train score) and delegates to ``_fit_stage``.
        Returns the number of stages fit; might differ from ``n_estimators`` due to early stopping.
        """
        n_samples   = X.shape[0]
        do_oob      = self.subsample < 1.0
        sample_mask = np.ones((n_samples, ), dtype=bool)
        n_inbag     = max(1, int(self.subsample * n_samples))
        loss_       = self.loss_

        if self.verbose:
            verbose_reporter = VerboseReporter(verbose=self.verbose)
            verbose_reporter.init(self, begin_at_stage)

        X_csc = csc_matrix(X) if issparse(X) else None
        X_csr = csr_matrix(X) if issparse(X) else None

        if self.n_iter_no_change is not None:
            loss_history = np.full(self.n_iter_no_change, np.inf)
            # We create a generator to get the predictions for X_val after the addition of each successive stage
            y_val_pred_iter = self._staged_raw_predict(X_val)

        # perform boosting iterations
        i = begin_at_stage
        for i in range(begin_at_stage, self.n_estimators):

            # subsampling
            if do_oob:
                sample_mask = _random_sample_mask(n_samples,
                                                  n_inbag,
                                                  random_state)

                # OOB score before adding this stage
                old_oob_score = loss_(y[~sample_mask],
                                      raw_predictions[~sample_mask],
                                      sample_weight[~sample_mask])

            # fit next stage of trees
            raw_predictions = self._fit_stage(i,
                                              X,
                                              y,
                                              raw_predictions,
                                              sample_weight,
                                              sample_mask,
                                              random_state,
                                              X_csc,
                                              X_csr)

            # track deviance (= loss)
            if do_oob:
                self.train_score_[i]     = loss_(y[sample_mask],
                                                 raw_predictions[sample_mask],
                                                 sample_weight[sample_mask])

                self.oob_improvement_[i] = old_oob_score - loss_(y[~sample_mask],
                                                                 raw_predictions[~sample_mask],
                                                                 sample_weight[~sample_mask])
            else:
                # no need to fancy index w/ no subsampling
                self.train_score_[i] = loss_(y, raw_predictions, sample_weight)

            if self.verbose > 0:
                verbose_reporter.update(i, self)

            if monitor is not None:
                early_stopping = monitor(i, self, locals())
                if early_stopping:
                    break

            # We also provide an early stopping based on the score from
            # validation set (X_val, y_val), if n_iter_no_change is set
            if self.n_iter_no_change is not None:
                # By calling next(y_val_pred_iter), we get the predictions for X_val after the addition of the current
                # stage
                validation_loss = loss_(y_val,
                                        next(y_val_pred_iter),
                                        sample_weight_val)

                # Require validation_score to be better (less) than at least one of the last n_iter_no_change evaluations
                if np.any(validation_loss + self.tol < loss_history):
                    loss_history[i % len(loss_history)] = validation_loss
                else:
                    break

        return i + 1

    def fit(self, X, y, sample_weight=None, monitor=None):
        """
        Fit the gradient boosting model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        y : array-like of shape (n_samples,)
            Target values (strings or integers in classification, real numbers
            in regression)
            For classification, labels must correspond to classes.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        monitor : callable, default=None
            The monitor is called after each iteration with the current
            iteration, a reference to the estimator and the local variables of
            ``_fit_stages`` as keyword arguments ``callable(i, self,
            locals())``. If the callable returns ``True`` the fitting procedure
            is stopped. The monitor can be used for various things such as
            computing held-out estimates, early stopping, model introspect, and
            snapshoting.

        Returns
        -------
        self : object
        """

        # if not warmstart - clear the estimator state
        if not self.warm_start:
            self._clear_state()

        # Check input
        # -------------------------------------------------------------------------------------------------------------

        # Since check_array converts both X and y to the same dtype, but the trees use different types for X and y,
        # checking them separately.
        X, y = self._validate_data(X,
                                   y,
                                   accept_sparse=['csr', 'csc', 'coo'],
                                   dtype=DTYPE,
                                   multi_output=True)

        n_samples, self.n_features_ = X.shape
        sample_weight_is_none       = sample_weight is None
        sample_weight               = _check_sample_weight(sample_weight, X)

        y = column_or_1d(y, warn=True)

        if is_classifier(self):
            y = self._validate_y(y, sample_weight)
        else:
            y = self._validate_y(y)

        # Partition data into training and validation sets
        if self.n_iter_no_change is not None:
            stratify = y if is_classifier(self) else None
            X, X_val, y, y_val, sample_weight, sample_weight_val = (train_test_split(X,
                                                                                     y,
                                                                                     sample_weight,
                                                                                     random_state=self.random_state,
                                                                                     test_size=self.validation_fraction,
                                                                                     stratify=stratify))
            if is_classifier(self):
                if self._n_classes != np.unique(y).shape[0]:
                    # We choose to error here. The problem is that the init
                    # estimator would be trained on y, which has some missing
                    # classes now, so its predictions would not have the
                    # correct shape.
                    raise ValueError('The training data after the early stopping split '
                                     'is missing some classes. Try using another random '
                                     'seed.')
        else:
            X_val = y_val = sample_weight_val = None

        # Check params
        # -------------------------------------------------------------------------------------------------------------
        self._check_params()

        # Base learner initialization and get raw_predictions
        # -------------------------------------------------------------------------------------------------------------
        if not self._is_initialized():
            # init state
            self._init_state()

            # fit initial model and initialize raw predictions
            if self.init_ == 'zero':
                raw_predictions = np.zeros(shape=(X.shape[0], self.loss_.K), dtype=np.float64)
            else:
                # XXX clean this once we have a support_sample_weight tag
                if sample_weight_is_none:
                    self.init_.fit(X, y)
                else:
                    msg = ("The initial estimator {} does not support sample "
                           "weights.".format(self.init_.__class__.__name__))
                    try:
                        self.init_.fit(X, y, sample_weight=sample_weight)

                    except TypeError as e:
                        # regular estimator without SW support
                        raise ValueError(msg) from e

                    except ValueError as e:
                        if "pass parameters to specific steps of "\
                           "your pipeline using the "\
                           "stepname__parameter" in str(e):  # pipeline
                            raise ValueError(msg) from e
                        else:  # regular estimator whose input checking failed
                            raise

                raw_predictions = self.loss_.get_init_raw_predictions(X, self.init_)

            begin_at_stage = 0

            # The rng state must be preserved if warm_start is True
            self._rng = check_random_state(self.random_state)

        else:
            # add more estimators to fitted model
            # invariant: warm_start = True
            if self.n_estimators < self.estimators_.shape[0]:
                raise ValueError('n_estimators=%d must be larger or equal to '
                                 'estimators_.shape[0]=%d when '
                                 'warm_start==True'
                                 % (self.n_estimators, self.estimators_.shape[0]))

            begin_at_stage = self.estimators_.shape[0]

            # The requirements of _decision_function (called in two lines below) are more constrained than fit.
            # It accepts only CSR matrices.
            X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
            raw_predictions = self._raw_predict(X)
            self._resize_state()

        # Fit the boosting stages
        # -------------------------------------------------------------------------------------------------------------
        n_stages = self._fit_stages(X,
                                    y,
                                    raw_predictions,
                                    sample_weight,
                                    self._rng,
                                    X_val,
                                    y_val,
                                    sample_weight_val,
                                    begin_at_stage,
                                    monitor)

        # Change shape of arrays after fitting (early-stopping or additional ests)
        # -------------------------------------------------------------------------------------------------------------
        if n_stages != self.estimators_.shape[0]:
            self.estimators_  = self.estimators_[:n_stages]
            self.train_score_ = self.train_score_[:n_stages]
            if hasattr(self, 'oob_improvement_'):
                self.oob_improvement_ = self.oob_improvement_[:n_stages]

        self.n_estimators_ = n_stages

        return self
