
import warnings
import numpy as np

from sklearn.base import RegressorMixin
from sklearn.utils.validation import _deprecate_positional_args, check_array, check_is_fitted
from sklearn.tree._tree import DTYPE, DOUBLE
from sklearn.exceptions import NotFittedError

from models.gradient_boosting_decision_trees.dp_gbdt import BaseDifferentiallyPrivateGradientBoosting


class DifferentiallyPrivateGradientBoostingRegressor(RegressorMixin, BaseDifferentiallyPrivateGradientBoosting):
    """
    Differentially Private Gradient Boosting for regression.

    GB builds an additive model in a forward stage-wise fashion;
    it allows for the optimization of arbitrary differentiable loss functions.
    In each stage a regression tree is fit on the negative gradient of the
    given loss function.

    Read more in the :ref:`User Guide <gradient_boosting>`.

    Parameters
    ----------
    loss : {'ls', 'lad', 'huber', 'quantile'}, default='ls'
        Loss function to be optimized. 'ls' refers to least squares
        regression. 'lad' (least absolute deviation) is a highly robust
        loss function solely based on order information of the input
        variables. 'huber' is a combination of the two. 'quantile'
        allows quantile regression (use `alpha` to specify the quantile).

    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.

    n_estimators : int, default=100
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.

    subsample : float, default=1.0
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.

    criterion : {'friedman_mse', 'mse', 'mae'}, default='friedman_mse'
        The function to measure the quality of a split. Supported criteria
        are "friedman_mse" for the mean squared error with improvement
        score by Friedman, "mse" for mean squared error, and "mae" for
        the mean absolute error. The default value of "friedman_mse" is
        generally the best as it can provide a better approximation in
        some cases.

        .. versionadded:: 0.18
        .. deprecated:: 0.24
            `criterion='mae'` is deprecated and will be removed in version
            1.1 (renaming of 0.26). The correct way of minimizing the absolute
            error is to use `loss='lad'` instead.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_depth : int, default=3
        Maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    min_impurity_split : float, default=None
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19. The default value of
           ``min_impurity_split`` has changed from 1e-7 to 0 in 0.23 and it
           will be removed in 1.0 (renaming of 0.25).
           Use ``min_impurity_decrease`` instead.

    init : estimator or 'zero', default=None
        An estimator object that is used to compute the initial predictions.
        ``init`` has to provide :term:`fit` and :term:`predict`. If 'zero', the
        initial raw predictions are set to zero. By default a
        ``DummyEstimator`` is used, predicting either the average target value
        (for loss='ls'), or a quantile for the other losses.

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to each Tree estimator at each
        boosting iteration.
        In addition, it controls the random permutation of the features at
        each split (see Notes for more details).
        It also controls the random spliting of the training data to obtain a
        validation set if `n_iter_no_change` is not None.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    max_features : {'auto', 'sqrt', 'log2'}, int or float, default=None
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    alpha : float, default=0.9
        The alpha-quantile of the huber loss function and the quantile
        loss function. Only if ``loss='huber'`` or ``loss='quantile'``.

    verbose : int, default=0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if ``n_iter_no_change`` is set to an integer.

        .. versionadded:: 0.20

    n_iter_no_change : int, default=None
        ``n_iter_no_change`` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside ``validation_fraction`` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous ``n_iter_no_change`` numbers of
        iterations.

        .. versionadded:: 0.20

    tol : float, default=1e-4
        Tolerance for the early stopping. When the loss is not improving
        by at least tol for ``n_iter_no_change`` iterations (if set to a
        number), the training stops.

        .. versionadded:: 0.20

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

        .. versionadded:: 0.22

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    oob_improvement_ : ndarray of shape (n_estimators,)
        The improvement in loss (= deviance) on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.
        Only available if ``subsample < 1.0``

    train_score_ : ndarray of shape (n_estimators,)
        The i-th score ``train_score_[i]`` is the deviance (= loss) of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the deviance on the training data.

    loss_ : LossFunction
        The concrete ``LossFunction`` object.

    init_ : estimator
        The estimator that provides the initial predictions.
        Set via the ``init`` argument or ``loss.init_estimator``.

    estimators_ : ndarray of DecisionTreeRegressor of shape (n_estimators, 1)
        The collection of fitted sub-estimators.

    n_classes_ : int
        The number of classes, set to 1 for regressors.

        .. deprecated:: 0.24
            Attribute ``n_classes_`` was deprecated in version 0.24 and
            will be removed in 1.1 (renaming of 0.26).

    n_estimators_ : int
        The number of estimators as selected by early stopping (if
        ``n_iter_no_change`` is specified). Otherwise it is set to
        ``n_estimators``.

    n_features_ : int
        The number of data features.

    max_features_ : int
        The inferred value of max_features.

    See Also
    --------
    HistGradientBoostingRegressor : Histogram-based Gradient Boosting
        Classification Tree.
    sklearn.tree.DecisionTreeRegressor : A decision tree regressor.
    sklearn.tree.RandomForestRegressor : A random forest regressor.

    Notes
    -----
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    """

    _SUPPORTED_LOSS = ('ls', 'lad', 'huber', 'quantile')

    @_deprecate_positional_args
    def __init__(self, *,
                 privacy_notion,
                 cat_idx             = None,
                 loss                = 'ls',
                 learning_rate       = 0.1,
                 n_estimators        = 100,
                 subsample           = 1.0,
                 min_samples_split   = 2,
                 max_depth           = 3,
                 init                = None,
                 random_state        = None,
                 verbose             = 0,
                 l2_threshold        = 1.0,
                 l2_lambda           = 0.1,
                 warm_start          = False,
                 validation_fraction = 0.1,
                 n_iter_no_change    = None,
                 tol                 = 1e-4):

        super().__init__(privacy_notion      = privacy_notion,
                         cat_idx             = cat_idx,
                         loss                = loss,
                         learning_rate       = learning_rate,
                         n_estimators        = n_estimators,
                         min_samples_split   = min_samples_split,
                         max_depth           = max_depth,
                         init                = init,
                         subsample           = subsample,
                         random_state        = random_state,
                         verbose             = verbose,
                         warm_start          = warm_start,
                         validation_fraction = validation_fraction,
                         n_iter_no_change    = n_iter_no_change,
                         tol                 = tol)

    def _validate_y(self, y, sample_weight=None):
        if y.dtype.kind == 'O':
            y = y.astype(DOUBLE)
        return y

    def _warn_mae_for_criterion(self):
        # TODO: This should raise an error from 1.1
        warnings.warn("criterion='mae' was deprecated in version 0.24 and "
                      "will be removed in version 1.1 (renaming of 0.26). The "
                      "correct way of minimizing the absolute error is to use "
                      " loss='lad' instead.", FutureWarning)

    def predict(self, X):
        """Predict regression target for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
        # In regression we can directly return the raw value from the trees.
        return self._raw_predict(X).ravel()

    def staged_predict(self, X):
        """Predict regression target at each stage for X.

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
        y : generator of ndarray of shape (n_samples,)
            The predicted value of the input samples.
        """
        for raw_predictions in self._staged_raw_predict(X):
            yield raw_predictions.ravel()

    def apply(self, X):
        """Apply trees in the ensemble to X, return leaf indices.

        .. versionadded:: 0.17

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will
            be converted to a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : array-like of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the ensemble,
            return the index of the leaf x ends up in each estimator.
        """

        leaves = super().apply(X)
        leaves = leaves.reshape(X.shape[0], self.estimators_.shape[0])
        return leaves

    # FIXME: to be removed in 1.1
    # mypy error: Decorated property not supported
    #@deprecated("Attribute n_classes_ was deprecated "  # type: ignore
    #            "in version 0.24 and will be removed in 1.1 "
    #            "(renaming of 0.26).")

    @property
    def n_classes_(self):
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError(
                "{} object has no n_classes_ attribute."
                .format(self.__class__.__name__)
            ) from nfe
        return 1
